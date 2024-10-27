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
    question = "What percentage of responses from Large Language Models are hallucinations?"
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


# Add these new functions to the existing code


def persona_based_prompting_example():
    """Demonstrates how to get responses tailored to different audience levels."""
    topic = "How does machine learning work?"
    personas = [
        "a 10-year-old student",
        "a business executive",
        "a computer science graduate",
    ]

    for persona in personas:
        prompt = f"""
        Explain {topic} to {persona}.
        Keep the explanation appropriate for their background and use relevant analogies 
        they would understand based on their context.
        """
        response = get_llm_response(prompt)
        print_prompt_and_response(f"Persona-Based ({persona})", prompt, response)


# Explanation: Persona-based prompting adapts the AI's response style and complexity to match
# different audience needs. This technique is effective because it ensures information is
# accessible and relevant to specific audiences. It's particularly useful in:
# - Educational settings with diverse student backgrounds
# - Business communications across different departments
# - Technical documentation for various stakeholder groups
# The key is specifying not just who the audience is, but also considering their context
# and likely frame of reference.


def template_based_prompting_example():
    """Shows how to use templates for consistent responses."""
    template = """
    Topic: {topic}
    Context: {context}
    
    Please analyze this topic using the following structure:
    1. Definition:
       - Simple explanation
       - Key components
    
    2. Applications:
       - Current use cases
       - Potential future uses
    
    3. Implications:
       - Benefits
       - Challenges
       - Ethical considerations
    
    4. Recommendations:
       - Best practices
       - Implementation guidelines
    """

    example = {
        "topic": "Facial recognition in public spaces",
        "context": "Growing adoption of surveillance technologies in smart cities",
    }

    prompt = template.format(**example)
    response = get_llm_response(prompt)
    print_prompt_and_response("Template-Based", prompt, response)


# Explanation: Template-based prompting uses predefined structures to ensure consistent
# and comprehensive responses. This technique is valuable because it:
# - Maintains consistency across multiple queries
# - Ensures all important aspects are covered
# - Makes responses easier to process and compare
# - Reduces the need for follow-up questions
# It's particularly useful in business settings where standardized analysis is needed,
# or in research contexts where systematic evaluation is important.


def comparative_analysis_prompting_example():
    """Demonstrates how to prompt for detailed comparisons."""
    prompt = """
    Compare and contrast the following AI technologies:
    1. Traditional Machine Learning
    2. Deep Learning
    3. Reinforcement Learning
    
    Use this structure:
    1. Key Characteristics
    2. Typical Applications
    3. Advantages
    4. Limitations
    5. Resource Requirements
    
    Present the comparison in a way that highlights the unique aspects of each approach
    and helps in understanding when to use which technology.
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Comparative Analysis", prompt, response)


# Explanation: Comparative analysis prompting helps in understanding relationships and
# trade-offs between different options. This technique is effective because it:
# - Forces systematic comparison across multiple dimensions
# - Highlights key differences and similarities
# - Helps in decision-making processes
# - Provides a comprehensive view of alternatives
# It's especially useful when evaluating technologies, methodologies, or approaches
# where understanding relative strengths and weaknesses is crucial.


def iterative_refinement_prompting_example():
    """Shows how to iteratively improve responses through multiple prompts."""
    initial_question = (
        "What are the key considerations for implementing AI in healthcare?"
    )

    # Initial broad response
    initial_prompt = f"Provide a high-level overview: {initial_question}"
    initial_response = get_llm_response(initial_prompt)

    # First refinement - focus on technical aspects
    technical_prompt = f"""
    Based on this overview:
    {initial_response}
    
    Deep dive into the technical implementation challenges and requirements.
    Focus on infrastructure, data management, and system integration.
    """
    technical_response = get_llm_response(technical_prompt)

    # Second refinement - focus on ethical considerations
    ethical_prompt = f"""
    Based on this overview:
    {initial_response}
    
    And given the technical considerations:
    {technical_response}
    
    Analyze the ethical implications and necessary safeguards.
    Address privacy, bias, and patient rights.
    """
    ethical_response = get_llm_response(ethical_prompt)

    print_prompt_and_response(
        "Iterative Refinement - Initial", initial_prompt, initial_response
    )
    print_prompt_and_response(
        "Iterative Refinement - Technical", technical_prompt, technical_response
    )
    print_prompt_and_response(
        "Iterative Refinement - Ethical", ethical_prompt, ethical_response
    )


# Explanation: Iterative refinement prompting uses a series of connected prompts to
# progressively develop more detailed and nuanced responses. This technique is powerful because:
# - It breaks complex topics into manageable chunks
# - Each iteration builds on previous responses
# - It allows for focused exploration of specific aspects
# - It helps maintain context while diving deeper
# This approach is particularly useful for complex topics where a single prompt
# might not capture all necessary details, or when different aspects need
# separate but connected analysis.


def scenario_based_prompting_example():
    """Demonstrates how to use realistic scenarios for context-rich responses."""
    scenario = """
    Scenario: MediCare Hospital is planning to implement an AI system for:
    1. Patient triage in the emergency department
    2. Predicting potential readmissions
    3. Scheduling staff based on predicted patient loads
    
    The hospital has:
    - 500 beds
    - 1,200 staff members
    - Electronic health records going back 5 years
    - A mix of urban and rural patients
    - Limited AI expertise on staff
    
    Required: Develop an implementation plan that addresses technical, 
    operational, and ethical considerations.
    """
    response = get_llm_response(scenario)
    print_prompt_and_response("Scenario-Based", scenario, response)


# Explanation: Scenario-based prompting provides rich context through realistic
# situations. This technique is effective because:
# - It grounds abstract concepts in concrete situations
# - It helps users understand practical applications
# - It forces consideration of real-world constraints
# - It makes responses more actionable and relevant
# This approach is particularly valuable in training contexts, case studies,
# and when helping users understand how to apply concepts in practice.
# The key is providing enough detail to make the scenario realistic while
# keeping it focused on the learning objectives.


# Verification examples


def self_verification_prompting_example():
    """Demonstrates advanced self-verification prompting technique."""
    problem = """
    In a class of 30 students:
    - 2/5 of the students play soccer
    - half of the students play basketball
    - a quarter of the students play both sports
    - 1/3 of the students play guitar and 2 of them play both sports and guitar
    
    How many students don't play either sport and play guitar?
    """

    verification_prompt = f"""
    Problem: {problem}
    
    Please solve this problem following these steps:
    1. First, provide your initial calculation
    2. Then verify your work by:
       - Breaking down each component
       - Checking your assumptions
       - Identifying potential errors
       - Recalculating using a different method
       - Comparing both results
    3. If you find any discrepancies, explain and correct them
    4. Critique the challenge of the problem if you think it's stated incorrectly
    5. Provide your final, verified answer
    
    Show all your work and reasoning. Readable text format, no code.
    """
    response = get_llm_response(verification_prompt)
    print_prompt_and_response("Self-Verification", verification_prompt, response)


def logical_verification_prompting_example():
    """Shows how to use verification through logical constraints."""
    query = """
    In a class of 30 students:
    - 2/5 of the students play soccer
    - half of the students play basketball
    - a quarter of the students play both sports
    - 1/3 of the students play guitar and 2 of them play both sports and guitar
    
    How many students don't play either sport and play guitar?
    """

    prompt = f"""
    Problem: {query}
    
    Please solve this problem using the following verification framework:
    
    1. Initial Solution:
       - Show your calculation
       - Explain your reasoning
    
    2. Verify Logical Constraints:
       - Total students must equal 30
       - Students in both sports ≤ minimum of individual sports
       - Total students in either sport ≤ sum of individual sports
       - Final answer cannot be negative
    
    3. Visual Verification:
       - Draw a Venn diagram (describe it in text)
       - Verify numbers match across all representations
    
    4. Edge Case Check:
       - What if all numbers were equal?
       - What if there was no overlap?
       - Do these cases reveal any issues in your logic?
       
    5. Critique the challenge of the problem if you think it's stated incorrectly
    
    6. Final Verified Answer:
       - State your conclusion
       - Confirm all constraints are satisfied
    """
    response = get_llm_response(prompt)
    print_prompt_and_response("Logical Verification", prompt, response)


# Explanation: Modern self-verification prompting has evolved beyond simple rechecking.
# It now incorporates multiple verification strategies:
# - Breaking down complex problems into verifiable components
# - Using different methods to cross-validate results
# - Applying logical constraints and edge cases
# - Leveraging visual representations for verification
# - Systematic error checking
#
# This technique is particularly effective because it:
# - Reduces errors in complex calculations
# - Helps identify hidden assumptions
# - Provides multiple perspectives on the same problem
# - Forces systematic thinking about edge cases
# - Creates more reliable and trustworthy outputs
#
# It's especially useful in:
# - Mathematical calculations
# - Logical reasoning problems
# - Data analysis tasks
# - Decision-making scenarios
# - Any situation where accuracy is crucial
#
# The key is to structure the verification process so it's not just
# repeating the same calculation, but approaching the problem from
# different angles to ensure robustness of the solution.


def join_list(items: List[str]) -> str:
    """Join list items with newlines, adding numbers."""
    return "\n".join(f"\n - {i+1}. {item}" for i, item in enumerate(items)) if items else ""


def generate_problem_solving_clarifications(
    questions: List[str] = [], answers: List[str] = []
) -> str:
    return f"""
    Problem to solve: 
      - 2/5 of the students play soccer
      - half of the students play basketball
      - a quarter of the students play both sports
      - 1/3 of the students play guitar and 2 of them play both sports and guitar
    How many students don't play either sport and play guitar?
    
    Your questions: 
    {join_list(questions)}
    
    My answers to your questions: 
    {join_list(answers)}

    Before solving this problem:
    1. First check if there are any ambiguities, missing information, or inconsistencies in the problem.
    
    2. If you find any issues:
       - Ask ONE clear, specific question to resolve the most critical ambiguity
       - Explain why this clarification is needed
       - Wait for an answer before proceeding further
       
    3. If the problem is completely clear:
       - State that no clarification is needed
       - Proceed with solving the problem step by step

    4. After receiving any clarification:
       - Acknowledge the clarification
       - Check if any other critical ambiguities remain
       - If clear, proceed with the solution
       - If not, ask the next most important question

    Remember: Ask only ONE question at a time, focusing on the most critical issue first.
    """


# Example usage
responses = []

prompt = generate_problem_solving_clarifications(None, None)
response = get_llm_response(prompt)
responses.append(response)

print_prompt_and_response("Problem Solving", prompt, response)

prompt = generate_problem_solving_clarifications(responses, ["Maybe"])
response = get_llm_response(prompt)
responses.append(response)

print_prompt_and_response("Problem Solving", prompt, response)


if __name__ == "__main__":
    client = initialize_openai_client()
    basic_prompt_example()
    # structured_prompt_example()
    # chain_of_thought_example()
    # few_shot_example()
    # role_playing_example()
    # task_decomposition_example()
    # zero_shot_example()
    # self_consistency_example()
    # constrained_generation_example()
    # socratic_method_example()
    # reflective_prompting_example()
    # guided_feedback_prompting_example()
    # persona_based_prompting_example()
    # template_based_prompting_example()
    # comparative_analysis_prompting_example()
    # iterative_refinement_prompting_example()
    # scenario_based_prompting_example()
    # self_verification_prompting_example()
    # logical_verification_prompting_example()
