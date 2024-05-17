import os
import re
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import json
import time
from groq_api import GroqAPI
from tavily import TavilyClient
import os

# Set up the Groq API client
api_key = os.getenv('GROQ_API_KEY', '')
client = GroqAPI(api_key=api_key)

# Define the models to use for each agent and their token limits
MODEL_MAX_TOKENS_PER_MIN = {
    "gemma-7b-it": 15000,
    "mixtral-8x7b-32768": 5000,
    "llama3-70b-8192": 6000,
    "llama3-8b-8192": 30000,
}

# Initialize the Rich Console
console = Console()

def opus_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    model = "llama3-8b-8192"
    token_limit = MODEL_MAX_TOKENS_PER_MIN[model]

    console.print(f"\n[bold]Calling Orchestrator for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))
    messages = [
        {
            "role": "system",
            "content": "You are an AI task orchestrator whose role is to break down high-level objectives into actionable sub-tasks. I will provide you with an objective, and your job is to analyze it and develop a plan for how to best accomplish that objective.\nPlease carefully consider this objective and identify the key components, milestones, or phases that would be involved in achieving it. Think about the specific steps that would need to be taken. Then, break the objective down into a series of clear, concrete sub-tasks. Each sub-task should be a distinct action that moves us closer to accomplishing the overall objective. The sub-tasks should be specific enough to be actionable and measurable. Collectively, completing all the sub-tasks should be sufficient to fully achieve the stated objective.\n After listing out the sub-tasks, please provide a brief justification inside <justification> tags explaining why you chose to break down the objective in this particular way. Explain how this set of sub-tasks effectively covers the full scope of the objective.\n Focus on just decomposing the objective into parts, not on actually completing the work to achieve the objective. The sub-tasks you output should be high-level milestones, not a complete to-do list of every granular action required.\nPlease carefully analyze the objective and put serious thought into the optimal sub-task breakdown before responding. Aim to be thorough, but keep your response reasonably concise."
        },
        {
            "role": "user",
            "content": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\\nFile content:\\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"
        }
    ]

    # Check if the file content is provided and contains any code blocks
    if file_content and isinstance(file_content, dict) and not any(code_content for code_content in file_content.values()):
        file_content = None
        console.print(Panel(f"Missing code content in file: {file_content}", title="[bold yellow]Missing Code Content[/bold yellow]", title_align="left", border_style="yellow"))
        # Generate a prompt for a sub-agent to handle missing code content
        missing_code_prompt = f"Please provide a prompt for a sub-agent to handle the missing code content in the following file:\n\n{file_content}"
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": missing_code_prompt}]
        })
    
    # Check for missing search query
    if use_search:
        messages[0]["content"].append({"type": "text", "text": "Please also generate a JSON object containing a single 'search_query' key, which represents a question that, when asked online, would yield important information for solving the subtask. The question should be specific and targeted to elicit the most relevant and helpful resources. Format your JSON like this, with no additional text before or after:\n{\"search_query\": \"<question>\"}\n"})

    opus_response = client.send_request({"model": model, "messages": messages, "max_tokens": 5000})

    if client.token_remaining < 100:
        client.wait_for_token_reset()

    response_text = opus_response["choices"][0]["message"]["content"]
    console.print(f" Total Tokens Used: {opus_response['usage']['completion_tokens']}")
    
    search_query = None
    if use_search:
        # Extract the JSON from the response
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group()
            try:
                search_query = json.loads(json_string)["search_query"]
                console.print(Panel(f"Search Query: {search_query}", title="[bold blue]Search Query[/bold blue]", title_align="left", border_style="blue"))
                response_text = response_text.replace(json_string, "").strip()
            except json.JSONDecodeError as e:
                console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
                console.print(Panel(f"Skipping search query extraction.", title="[bold yellow]Search Query Extraction Skipped[/bold yellow]", title_align="left", border_style="yellow"))
        else:
            search_query = None

    console.print(Panel(response_text, title=f"[bold green]Groq Orchestrator[/bold green]", title_align="left", border_style="green", subtitle="Sending task to Subagent 👇"))
    return response_text, file_content, search_query

def haiku_sub_agent(prompt, search_query=None, previous_haiku_tasks=None, use_search=False, continuation=False):
    if previous_haiku_tasks is None:
        previous_haiku_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    system_message = "Previous Haiku tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_haiku_tasks)
    if continuation:
        prompt = continuation_prompt

    qna_response = None
    if search_query and use_search:
        # Initialize the Tavily client
        tavily = TavilyClient(api_key="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")
        # Perform a QnA search based on the search query
        qna_response = tavily.qna_search(query=search_query)
        console.print(f"QnA response: {qna_response}", style="yellow")

    # Prepare the messages array with only the prompt initially
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if qna_response:
        messages[0]["content"].append({"type": "text", "text": f"\nSearch Results:\n{qna_response}"})

    # messages = [
    #     {
    #         "role": "system",
    #         "content": system_message
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt
    #     }
    # ]

    haiku_response = client.send_request({"model": "mixtral-8x7b-32768", "messages": messages, "max_tokens": 4000})

    if client.token_remaining < 100:
        client.wait_for_token_reset()

    response_text = haiku_response["choices"][0]["message"]["content"]
    console.print(f"Input Tokens: {haiku_response['usage']['prompt_tokens']}, Output Tokens: {haiku_response['usage']['completion_tokens']}")
    
    completiontokens = haiku_response['usage']['completion_tokens']

    if completiontokens >= 4000:  # Threshold set to 4000 as a precaution
        console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
        continuation_response_text = haiku_sub_agent(prompt, search_query, previous_haiku_tasks, use_search, continuation=True)
        response_text += continuation_response_text

    console.print(Panel(response_text, title="[bold blue]Groq Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to Orchestrator 👇"))
    return response_text

def opus_refine(objective, sub_task_results, filename, projectname, continuation=False):
    console.print("\nCalling Opus to provide the refined final output for your objective:")
    messages = [
        {
            "role": "system",
            "content": "Your task is to take a set of subtask results and refine them into a high-quality final output that addresses an overall task description.\nPlease carefully review the task description to make sure you have a clear understanding of the ultimate goal that needs to be achieved.\nNext, critically examine each of the provided subtask results. Evaluate the quality and relevance of each one - how well does it contribute to addressing the overall task? Also analyze how well the subtask results fit together. Are they coherent and aligned with each other, or are there any contradictions or inconsistencies across the different results?\n<reflection>Think through how you can synthesize the subtask results together into a comprehensive final output. Consider:\n- What is the most logical way to structure and present the information from the subtasks? \n- How can you ensure a natural flow and transition between the different pieces while avoiding redundancy? \n- Is there any important information missing from the subtasks that needs to be added?\n- Does the compiled output fully address all aspects of the overall task description? \n</reflection>\nOnce you have a clear plan, generate the final refined output. Make sure it is well-organized, easy to follow, and covers the task description completely. Aim to create a polished, coherent and standalone final work product.\nPlease provide your final output inside <result> tags."
        },
        {
            "role": "user",
            "content": f"Here is the Objective you are trying to achieve: " + objective + "\n\nHere are the Sub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. Make sure the code files are completed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name in this format 'Filename: <filename>' NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n​python\n<code>\n"
        }
    ]

    if continuation:
        messages[1]["content"] += "\n\nContinue from the previous output."

    opus_response = client.send_request({"model": "llama3-70b-8192", "messages": messages, "max_tokens": 5000})

    if client.token_remaining < 100:
        client.wait_for_token_reset()

    response_text = opus_response["choices"][0]["message"]["content"]
    console.print(f"Input Tokens: {opus_response['usage']['prompt_tokens']}, Output Tokens: {opus_response['usage']['completion_tokens']}")

    opus_completiontokens = opus_response['usage']['completion_tokens']
    
    if opus_completiontokens >= 5000 and not continuation:  # Threshold set to 4000 as a precaution
        console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
        continuation_response_text = opus_refine(objective, sub_task_results + [response_text], filename, projectname, continuation=True)
        response_text += "\n" + continuation_response_text
    
    console.print(Panel(response_text, title="[bold green]Opus Refined Output[/bold green]", title_align="left", border_style="green", subtitle="Refinement complete"))
    return response_text

if __name__ == "__main__":
    def create_folder_structure(project_name, folder_structure, code_blocks):
        # Create the project folder
        try:
            os.makedirs(project_name, exist_ok=True)
            console.print(Panel(f"Created project folder: [bold]{project_name}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
        except OSError as e:
            console.print(Panel(f"Error creating project folder: [bold]{project_name}[/bold]\nError: {e}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
            return

        # Recursively create the folder structure and files
        create_folders_and_files(project_name, folder_structure, code_blocks)

    def create_folders_and_files(current_path, structure, code_blocks):
        for key, value in structure.items():
            path = os.path.join(current_path, key)
            if isinstance(value, dict):
                try:
                    os.makedirs(path, exist_ok=True)
                    console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                    create_folders_and_files(path, value, code_blocks)
                except OSError as e:
                    console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
            else:
                code_content = next((code for file, code in code_blocks if file == key), None)
                if code_content:
                    try:
                        with open(path, 'w') as file:
                            file.write(code_content)
                        console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
                    except IOError as e:
                        console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))
                else:
                    console.print(Panel(f"Code content not found for file: [bold]{key}[/bold]", title="[bold yellow]Missing Code Content[/bold yellow]", title_align="left", border_style="yellow"))

    def read_file(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    # Ask the user if they want to include any files in the objective
    include_files = input("Do you want to include any files in the objective? (y/N) ").lower() in ['y', 'yes']
    file_paths = []
    if include_files:
        # Ask the user for the number of files
        num_files = int(input("Enter the number of files: "))

        for i in range(num_files):
            file_path = input(f"Enter the file path for file {i+1} (or press Enter to finish): ")
            if file_path:
                file_paths.append(file_path)
            else:
                break
    
    objective = input("Please enter your objective: ")

    # Check if the input contains a file path
    file_content = None
    if "./" in objective or "/" in objective:
        file_path_match = re.findall(r'[./\w]+\.[\w]+', objective)
        if file_path_match:
            file_path = file_path_match[0]
            # Read the file content
            with open(file_path, 'r') as file:
                file_content = file.read()
            # Update the objective string to remove the file path
            objective = objective.split(file_path)[0].strip()

    # Ask the user if they want to use search
    use_search = input("Do you want to use search? (y/n): ").lower() == 'y'
    task_exchanges = []
    haiku_tasks = []
    while True:
        # Call Orchestrator to break down the objective into the next sub-task or provide the final output
        previous_results = [result for _, result in task_exchanges]
        if not task_exchanges:
            # Pass the file content only in the first iteration if available
           opus_result, file_content_for_haiku, search_query = opus_orchestrator(objective, file_content, previous_results, use_search)
        else:
            opus_result, _, search_query = opus_orchestrator(objective, previous_results=previous_results, use_search=use_search)

        if "The task is complete:" in opus_result:
            # If Opus indicates the task is complete, exit the loop
            final_output = opus_result.replace("The task is complete:", "").strip()
            break
        else:
            sub_task_prompt = opus_result
            # Append file content to the prompt for the initial call to haiku_sub_agent, if applicable
            if file_content_for_haiku and not haiku_tasks:
                sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_haiku}"
            # Call haiku_sub_agent with the prepared prompt and record the result
            sub_task_result = haiku_sub_agent(sub_task_prompt, search_query, haiku_tasks, use_search)
            # Log the task and its result for future reference
            haiku_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
            # Record the exchange for processing and output generation
            task_exchanges.append((sub_task_prompt, sub_task_result))
            # Prevent file content from being included in future haiku_sub_agent calls
            file_content_for_haiku = None

    # Create the .md filename
    sanitized_objective = re.sub(r'\W+', '_', objective)
    timestamp = datetime.now().strftime("%H-%M-%S")
    # Call Opus to review and refine the sub-task results
    refined_output = opus_refine(objective, [result for _, result in task_exchanges], timestamp, sanitized_objective)

    # Extract the project name from the refined output
    project_name_match = re.search(r'Project Name: (.*)', refined_output)
    project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

    # Extract the folder structure from the refined output
    folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
    folder_structure = {}
    if folder_structure_match:
        json_string = folder_structure_match.group(1).strip()
        try:
            folder_structure = json.loads(json_string)
        except json.JSONDecodeError as e:
            console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
            console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))

    # Extract code files from the refined output
    code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)

    # Create the folder structure and code files
    create_folder_structure(project_name, folder_structure, code_blocks)

    # Truncate the sanitized_objective to a maximum of 50 characters
    max_length = 25
    truncated_objective = sanitized_objective[:max_length] if len(sanitized_objective) > max_length else sanitized_objective

    # Update the filename to include the project name
    filename = f"{timestamp}_{truncated_objective}.md"

    # Prepare the full exchange log
    exchange_log = f"Objective: {objective}\n\n"
    exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
    for i, (prompt, result) in enumerate(task_exchanges, start=1):
        exchange_log += f"Task {i}:\n"
        exchange_log += f"Prompt: {prompt}\n"
        exchange_log += f"Result: {result}\n\n"

    exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
    exchange_log += refined_output

    console.print(f"\n[bold]Refined Final output:[/bold]\n{refined_output}")

    with open(filename, 'w') as file:
        file.write(exchange_log)
    print(f"\nFull exchange log saved to {filename}")
