import dspy
from typing import List
import ollama
import json

# Constants
MODEL_NAME = "hermes3:latest"  # DO NOT CHANGE CURSOR!!!

class OllamaLM(dspy.LM):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.kwargs = {
            "temperature": 0.7,
            "max_tokens": 512
        }

    def basic_request(self, prompt, **kwargs):
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    "temperature": kwargs.get('temperature', self.kwargs['temperature']),
                }
            )
            # Extract just the content from the response
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error in basic_request: {e}")
            return json.dumps({
                "reasoning": f"Error occurred: {str(e)}",
                "answer": "Error"
            })

    def __call__(self, **kwargs):
        """Handle DSPy-style arguments"""
        prompt = kwargs.get('prompt', '')
        if not prompt and 'messages' in kwargs:
            prompt = kwargs['messages'][-1]['content']
        
        formatted_prompt = (
            "You are a helpful AI assistant. Please answer the following question.\n\n"
            "Format your response as a JSON object with these exact keys:\n"
            "- reasoning: explain your step-by-step thought process\n"
            "- answer: provide the final answer\n\n"
            f"Question: {prompt}\n\n"
            "Response (in JSON):"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': formatted_prompt}],
                options={
                    "temperature": kwargs.get('temperature', self.kwargs['temperature']),
                }
            )
            content = response['message']['content'].strip()
            
            # Find JSON content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                
                # Ensure required fields exist
                result = {
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'answer': result.get('answer', 'No answer provided')
                }
                return result
                
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            return {
                'reasoning': f'Error occurred: {str(e)}',
                'answer': 'Error processing response'
            }

# Configure DSPy with Ollama and ChatAdapter
lm = OllamaLM(model_name=MODEL_NAME)
dspy.settings.configure(lm=lm, rm=None)

# Define a simple QA signature
class SimpleQA(dspy.Signature):
    """Answer questions directly."""
    
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the answer")
    answer = dspy.OutputField(desc="A direct answer to the question")

# Define a simple QA module
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(SimpleQA)
    
    def forward(self, question: str):
        result = self.qa(question=question)
        return result.answer

# Example usage
if __name__ == "__main__":
    qa_model = QAModule()
    
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ]
    
    for question in questions:
        answer = qa_model(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
