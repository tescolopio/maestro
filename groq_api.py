'''
------------------------------------------------------------------------------
Changelog
Date | Name | Description | Tag
2024-05-16.1 | Tim | Add code to repo | 20240516TE
------------------------------------------------------------------------------
'''

import time
import random
import requests
import logging
import tkinter as tk
from tkinter import ttk

class GroqAPI:
    def __init__(self, api_key, max_retries=10, base_delay=1):
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.token_counters = {
            "llama3-70b-8192": {"tokens": 0, "requests": 0, "start_time": None},
            "llama3-8b-8192": {"tokens": 0, "requests": 0, "start_time": None},
            "gemma-7b-it": {"tokens": 0, "requests": 0, "start_time": None},
            "mixtral-8x7b-32768": {"tokens": 0, "requests": 0, "start_time": None},
        }
        self.rate_limits = {
            "llama3-70b-8192": {"tokens_per_min": 6000, "requests_per_day": 14400},
            "llama3-8b-8192": {"tokens_per_min": 30000, "requests_per_day": 14400},
            "gemma-7b-it": {"tokens_per_min": 15000, "requests_per_day": 14400},
            "mixtral-8x7b-32768": {"tokens_per_min": 5000, "requests_per_day": 14400},
        }
        self.last_minute_check = time.time()
        self.root = None
        self.token_labels = {}

    def start_gui(self):
        self.root = tk.Tk()
        self.root.title("Token Usage Monitor")
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        for i, model in enumerate(self.token_counters.keys()):
            ttk.Label(frame, text=model).grid(column=0, row=i, sticky=tk.W)
            self.token_labels[model] = ttk.Label(frame, text="Tokens: 0, Requests: 0")
            self.token_labels[model].grid(column=1, row=i, sticky=tk.W)

        self.update_gui()
        self.root.mainloop()
        # self.token_limit = None
        # self.token_remaining = None
        # self.token_reset = None
        
    def update_gui(self):
        for model, data in self.token_counters.items():
            tokens = data['tokens']
            requests = data['requests']
            self.token_labels[model].config(text=f"Tokens: {tokens}, Requests: {requests}")
        self.root.after(1000, self.update_gui)

    def send_request(self, request):
        response = None
        for i in range(self.max_retries):
            try:
                response = self.send_request_to_api(request)
                self.update_rate_limits(response)
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response is not None and response.status_code == 429:
                    retry_after = int(response.headers.get('retry-after', 1))
                    self.logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                else:
                    delay = self.calculate_delay(i)
                    self.logger.error(f"Error: {e}. Retrying in {delay} seconds.")
                    if response is not None:
                        self.logger.error(f"Status Code: {response.status_code}")
                        self.logger.error(f"Response Headers: {response.headers}")
                        self.logger.error(f"Response Content: {response.text}")
                    time.sleep(delay)
                    if i == self.max_retries - 1:
                        self.logger.critical("Maximum retries exceeded.")
                        raise Exception("Maximum retries exceeded.")
        return None

    def calculate_delay(self, attempt):
        delay = self.base_delay * 2 ** attempt
        return delay

    def send_request_to_api(self, request):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=request)
        response.raise_for_status()
        return response

    def update_rate_limits(self, response):
        headers = response.headers
        self.token_limit = int(headers.get('x-ratelimit-limit-tokens', 0))
        self.token_remaining = int(headers.get('x-ratelimit-remaining-tokens', 0))
        self.token_reset = float(headers.get('x-ratelimit-reset-tokens', '0').replace('s', ''))

    def wait_for_token_reset(self):
        if self.token_reset:
            self.logger.info(f"Waiting for {self.token_reset} seconds to reset token limits.")
            time.sleep(self.token_reset)
