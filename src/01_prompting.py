from openai import OpenAI
from typing import List
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Define model constants
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

client: OpenAI


def initialize_openai_client() -> OpenAI:
    """Initialize and return an OpenAI client."""
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


# Function to get response based on the selected model
def get_llm_response(prompt: str, system_prompt: str = None) -> str:
    """Get a response from OpenAI's chat model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant for the minor AI for Society.",
            }
        )
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content


def get_embedding(text: str) -> List[float]:
    """Get an embedding from the embedding model."""
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


# Helper function to print prompt and response
def print_prompt_and_response(technique_name: str, prompt: str, response: str):
    print(f"\n{'-'*20} {technique_name} {'-'*20}")
    print(f"Prompt:\n{prompt}")
    print(f"\nResponse:\n{response}")
    print(f"{'-'*50}\n")


# Various prompting techniques


def basic_prompt_example():
    question = "What are the potential impacts of AI on privacy in smart cities?"
    response = get_llm_response(question)
    print_prompt_and_response("Basic Prompt", question, response)


# Explanation: Basic prompting is the simplest form of interaction with an AI model.
# It works well for straightforward questions because it allows the model to use its
# general knowledge without constraints. However, it may lack specificity or structure
# in complex scenarios.


def structured_prompt_example():
    question = "How can AI be used to improve sustainability in urban environments?"
    context = "Smart cities are increasingly using AI for resource management."
    format_requirements = {
        "application": "string",
        "benefits": "list",
        "challenges": "list",
    }
    evaluation_criteria = ["feasibility", "ethical considerations", "societal impact"]

    prompt = f"""
    Context: {context}
    
    Question: {question}
    
    Required Format:
    {json.dumps(format_requirements, indent=2)}
    
    Evaluation Criteria:
    {' '.join(f'- {criterion}' for criterion in evaluation_criteria)}
    
    Provide your response following exactly the format above, considering the ethical implications and societal impact.
    """

    response = get_llm_response(prompt)
    print_prompt_and_response("Structured Prompt", prompt, response)


# Explanation: Structured prompting provides a clear format and expectations for the AI's response.
# This technique works well because it guides the model to organize information in a specific way,
# ensuring that all required elements are addressed. It's particularly useful for complex queries
# that require a systematic approach.


def chain_of_thought_example():
    question = (
        "What are the ethical considerations of using AI in healthcare diagnostics?"
    )
    prompt = f"""
    Question: {question}
    Let's approach this step-by-step:
    1) First, let's consider...
     2) Next, we should think about...
    3) Finally, we can conclude...

    Now, based on this reasoning, the answer is:
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Chain of Thought", prompt, response)


# Explanation: Chain of Thought prompting encourages the AI to break down its reasoning process.
# This technique is effective because it mimics human problem-solving, allowing for more transparent
# and logical responses. It's particularly useful for complex questions that require step-by-step analysis.


def few_shot_example():
    question = "What is the capital of Australia?"
    prompt = f"""
    Here are a few examples:
    Q: What is the capital of Spain?
    A: The capital of Spain is Madrid.

    Q: What is the largest planet in our solar system?
    A: The largest planet in our solar system is Jupiter.

    Now, please answer the following question:
    Q: {question}
    A:
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Few-Shot Learning", prompt, response)


# Explanation: Few-shot learning provides the AI with examples before asking the main question.
# This technique works well because it gives the model a pattern to follow, improving consistency
# and accuracy in responses. It's especially useful when you want the AI to mimic a specific
# response style or format.


def role_playing_example():
    scenario = "A city is considering implementing AI-powered facial recognition in public spaces."
    role = "privacy advocate"
    system_prompt = f"You are an expert {role} working on AI applications in society. Provide responses from this perspective, considering ethical implications and societal impact."
    user_prompt = f"""
    Given the following scenario, provide your professional opinion and advice:

    Scenario: {scenario}

    Please provide your response in a professional manner, drawing from your expertise and addressing potential societal consequences.
    """
    response = get_llm_response(user_prompt, system_prompt=system_prompt)
    print_prompt_and_response(
        "Role-Playing", f"System: {system_prompt}\n\nUser: {user_prompt}", response
    )


# Explanation: Role-playing prompts the AI to adopt a specific persona or expertise.
# This technique is effective because it narrows the model's focus to a particular perspective,
# leading to more specialized and contextually appropriate responses. It's useful for getting
# expert-like opinions on specific topics.


def task_decomposition_example():
    complex_task = "Develop a plan to reduce carbon emissions in a major city using AI technologies."
    prompt = f"""
    Complex Task: {complex_task}

    Let's break this task down into smaller, manageable steps:

    1) [First step]
    2) [Second step]
    3) [Third step]
    ...

    Now, let's go through each step:

    1) [Explanation of first step]
    2) [Explanation of second step]
    3) [Explanation of third step]
    ...

    Conclusion: [Summary of the approach and final answer]
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Task Decomposition", prompt, response)


# Explanation: Task decomposition breaks down complex problems into smaller, manageable steps.
# This technique works well because it helps the AI (and the user) approach complex issues
# systematically. It's particularly useful for planning or strategizing tasks that involve
# multiple components or stages.


def zero_shot_example():
    question = "How can AI be used to improve traffic management in smart cities?"
    prompt = f"""
    Question: {question}
    
    Please provide a detailed answer without any additional context or examples.
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Zero-Shot Learning", prompt, response)


# Explanation: Zero-shot learning tests the AI's ability to answer questions without specific examples.
# This technique works because it leverages the model's general knowledge and ability to understand
# context. It's useful for assessing the AI's baseline capabilities on various topics.


def self_consistency_example():
    question = "What are the potential impacts of AI on job markets in the next decade?"
    prompt = f"""
    Question: {question}
    
    Please provide three different perspectives on this issue:
    
    Perspective 1:
    [Your response here]
    
    Perspective 2:
    [Your response here]
    
    Perspective 3:
    [Your response here]
    
    Now, synthesize these perspectives into a coherent answer:
    [Your synthesized response here]
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Self-Consistency", prompt, response)


# Explanation: Self-consistency prompts the AI to generate multiple perspectives and then synthesize them.
# This technique is effective because it encourages a more comprehensive and balanced approach to a topic.
# It's particularly useful for complex issues where multiple viewpoints need to be considered.


def constrained_generation_example():
    scenario = "A city is implementing an AI-powered traffic light system."
    prompt = f"""
    Scenario: {scenario}
    
    Please provide a response that meets the following constraints:
    1. Use exactly 50 words.
    2. Include at least one potential benefit and one potential risk.
    3. End with a question for further consideration.
    
    Your response:
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Constrained Generation", prompt, response)


# Explanation: Constrained generation sets specific limits or requirements for the AI's response.
# This technique works well because it challenges the AI to be creative within defined boundaries,
# often resulting in more focused and precise outputs. It's useful when you need responses in a
# very specific format or style.


def socratic_method_example():
    topic = "The impact of social media on mental health"
    initial_question = "What are the primary ways social media affects mental health?"

    questions = [initial_question]
    answers = []

    for i in range(3):  # Let's do 3 rounds of questions
        current_question = questions[-1]
        prompt = f"""
        Topic: {topic}
        Previous questions: {questions[:-1]}
        Previous answers: {answers}
        
        Current question: {current_question}
        
        Please provide a concise answer to the current question, and then ask a follow-up question 
        that digs deeper into the topic, challenging assumptions or exploring implications of the answer.
        
        Format your response as:
        Answer: [Your answer here]
        Follow-up question: [Your follow-up question here]
        """

        response = get_llm_response(prompt)
        print_prompt_and_response(f"Socratic Method - Round {i+1}", prompt, response)

        # Parse the response to get the answer and next question
        answer, next_question = response.split("Follow-up question:")
        answers.append(answer.strip().replace("Answer: ", ""))
        questions.append(next_question.strip())


# Explanation: The Socratic method uses a series of questions to stimulate critical thinking and illuminate ideas.
# This technique is effective because it encourages the AI to explore a topic deeply, considering multiple angles
# and uncovering underlying assumptions. It's particularly useful for complex topics that benefit from
# a dialogue-like exploration and for generating insightful questions for further investigation.


def reflective_prompting_example():
    initial_question = "How can AI be used to address climate change?"
    prompt = f"""
    Initial question: {initial_question}

    Please provide an initial response to this question. Then, reflect on your answer by considering:
    1. What assumptions did you make in your response?
    2. Are there any potential biases in your answer?
    3. What additional information would be helpful to provide a more comprehensive response?
    4. How might different stakeholders view this solution differently?

    After reflection, provide an updated, more nuanced response.

    Format your answer as follows:
    Initial response: [Your initial response here]

    Reflection:
    1. Assumptions: [List assumptions]
    2. Potential biases: [Discuss potential biases]
    3. Additional information needed: [List additional information]
    4. Stakeholder perspectives: [Discuss different viewpoints]

    Updated response: [Your updated, more nuanced response here]
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Reflective Prompting", prompt, response)


# Explanation: Reflective prompting encourages the AI to critically examine its own responses.
# This technique is effective because it promotes a more thoughtful and comprehensive analysis,
# addressing potential shortcomings in the initial response. It's particularly useful for
# complex topics where considering multiple perspectives and acknowledging limitations is important.


def guided_feedback_prompting_example():
    initial_question = (
        "What are the potential impacts of AI on job markets in the next decade?"
    )
    prompt = f"""
    Initial question: {initial_question}

    Please provide an initial response to this question.
    """
    initial_response = get_llm_response(prompt)

    feedback_prompt = f"""
    Your initial response was:
    {initial_response}

    This response is a good start, but let's improve it with the following guidance:
    1. Consider more diverse scenarios, including edge cases.
    2. Discuss both short-term and long-term risks.
    3. Include potential mitigation strategies for each risk.
    4. Address the ethical implications of these risks.

    Please respond by responding to the feedback then providing an updated response incorporating this feedback.
    """
    updated_response = get_llm_response(feedback_prompt)

    print_prompt_and_response(
        "Guided Feedback Prompting",
        f"{feedback_prompt[:300]}...\n\nUpdated Response after feedback:",
        updated_response,
    )


# Explanation: Guided feedback prompting involves providing specific feedback or guidance to the AI
# after its initial response. This technique allows the AI to refine and improve its answer based on
# targeted suggestions. It's particularly useful for obtaining more comprehensive, nuanced responses
# and for directing the AI's focus to specific aspects of a complex topic.


# ... (continue with other prompting techniques)

if __name__ == "__main__":
    client = initialize_openai_client()
    basic_prompt_example()
    structured_prompt_example()
    chain_of_thought_example()
    few_shot_example()
    role_playing_example()
    task_decomposition_example()
    zero_shot_example()
    self_consistency_example()
    constrained_generation_example()
    socratic_method_example()
    reflective_prompting_example()
    guided_feedback_prompting_example()
