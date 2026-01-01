import json
import os
import re
from typing import List, Dict, Optional
from message import LangChainMessageAPI
from prompt import (
    CHAIN_OF_THOUGHT_TEMPLATE,
    TRAINING_PROMPT_TEMPLATE,
    ROLE_RESPONSE_PROMPT_TEMPLATE,
    EVALUATION_PROMPT_TEMPLATE,
    ETHICAL_EVALUATION_SYSTEM_PROMPT,
    ETHICAL_EVALUATION_TEMPLATE,
    get_role_system_prompt
)

class RoleSafetyAnalyzer:
    """Tool for analyzing role-sensitive ethical risks with LLM integration"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.examples = []
        self.llm_api = None  # Will be initialized on demand
        self.load_data()
    
    def load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Error: File not found {self.data_path}")
            return
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} examples")
    
    def show_statistics(self):
        """Display data statistics"""
        print("\n" + "=" * 70)
        print("Data Statistics")
        print("=" * 70)
        
        # Role distribution
        roles = {}
        risk_categories = {}
        
        for ex in self.examples:
            role = ex.get('role', 'unknown')
            risk = ex.get('risk_category', 'unknown')
            
            roles[role] = roles.get(role, 0) + 1
            risk_categories[risk] = risk_categories.get(risk, 0) + 1
        
        print(f"\nRole Distribution:")
        for role, count in roles.items():
            print(f"  • {role}: {count}")
        
        print(f"\nRisk Categories:")
        for risk, count in risk_categories.items():
            print(f"  • {risk}: {count}")
    
    def generate_chain_of_thought_prompt(self, example: Dict) -> str:
        """Generate chain-of-thought prompt for each example"""
        return CHAIN_OF_THOUGHT_TEMPLATE.format(
            role=example['role'],
            question=example['question'],
            risk_category=example['risk_category'],
            harmful_response=example['harmful_response'],
            safety_anchored_response=example['safety_anchored_response']
        )
    
    def _initialize_llm_api(self):
        """Initialize LLM API on first use"""
        if self.llm_api is None:
            self.llm_api = LangChainMessageAPI()
        return self.llm_api
    
    def load_evaluation_questions(self, eval_path: str = "AsiaCement/evaluation.jsonl") -> List[Dict]:
        """Load evaluation questions from jsonl file"""
        questions = []
        if not os.path.exists(eval_path):
            print(f"Error: Evaluation file not found at {eval_path}")
            return questions
        
        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        return questions
    
    def generate_training_data(self, output_path: str):
        """Generate chain-of-thought training data"""
        
        training_examples = []
        
        for ex in self.examples:
            cot_prompt = self.generate_chain_of_thought_prompt(ex)
            
            training_examples.append({
                "role": ex['role'],
                "question": ex['question'],
                "chain_of_thought_prompt": cot_prompt,
                "harmful_response": ex['harmful_response'],
                "safety_anchored_response": ex['safety_anchored_response'],
                "risk_category": ex['risk_category']
            })
        
        # Save training data
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Generated {len(training_examples)} training examples")
        print(f"✓ Saved to: {output_path}")
    
    def get_role_system_prompt(self, index: int) -> str:
        """Get system prompt for role-playing as the character"""
        if index >= len(self.examples):
            return None
        
        example = self.examples[index]
        role = example['role']
        return get_role_system_prompt(role)
    
    def train_role_actor(self, index: int) -> Dict:
        """Train an actor to play the role and have them generate a response.

        Uses all examples of the same role as few-shot guidance to keep role fidelity.
        """
        if index >= len(self.examples):
            return {"success": False, "error": f"Index {index} out of range"}
        
        api = self._initialize_llm_api()
        example = self.examples[index]
        role = example['role']
        
        # Collect all examples sharing the same role for context
        role_examples = [ex for ex in self.examples if ex.get('role') == role]
        context_blocks = []
        for i, ex in enumerate(role_examples, 1):
            context_blocks.append(
                f"Example {i} - Question: {ex['question']}\n"
                f"High-Risk Response (do not copy): {ex['harmful_response']}\n"
                f"Safety-Anchored Response (preferred pattern): {ex['safety_anchored_response']}"
            )
        role_context = "\n\n".join(context_blocks)
        
        # Set role-specific system prompt
        system_prompt = self.get_role_system_prompt(index)
        api.set_system_message(system_prompt)
        
        # Ask the actor to respond to the question with in-role consistency
        role_response_prompt = ROLE_RESPONSE_PROMPT_TEMPLATE.format(
            question=example['question'],
            role_context=role_context
        )
        
        # Invoke LLM to generate role response
        result = api.invoke(role_response_prompt, metadata={"example_index": index, "type": "role_response"})
        return result
    
    def evaluate_ethical_alignment(
        self,
        actor_response: str,
        role: str,
        question: str,
        risk_category: str = "unspecified",
        harmful_response: str = "N/A",
        safety_anchored_response: str = "N/A",
    ) -> Dict:
        """Evaluate whether an actor's response is ethically sound.

        This uses a fresh LLM instance so it does not disturb the role-playing
        system prompt used during response generation.
        """
        eval_model = os.getenv("EVAL_MODEL_NAME")
        api = LangChainMessageAPI(model=eval_model)
        # Set system prompt for ethical evaluation
        api.set_system_message(ETHICAL_EVALUATION_SYSTEM_PROMPT)

        # Create evaluation prompt
        evaluation_prompt = ETHICAL_EVALUATION_TEMPLATE.format(
            role=role,
            question=question,
            actor_response=actor_response,
            risk_category=risk_category,
            harmful_response=harmful_response,
            safety_anchored_response=safety_anchored_response
        )

        # Invoke LLM for evaluation
        result = api.invoke(
            evaluation_prompt,
            metadata={"type": "ethical_evaluation", "role": role, "risk_category": risk_category}
        )
        return result

    def _extract_conclusion(self, text: str) -> str:
        """Extract conclusion section if present; fallback to last paragraph."""
        if not text:
            return ""

        # Look for a 'Conclusion:' header (case-insensitive)
        match = re.search(r"(?i)conclusion[:\s]*\n?(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Otherwise, take the last non-empty paragraph
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if parts:
            return parts[-1]

        return text.strip()
    
    def _print_example_choices(self):
        """Print available examples with indices for selection."""
        print("\nAvailable examples:")
        for i, ex in enumerate(self.examples):
            question_preview = ex['question'][:80].replace('\n', ' ')
            print(f"  {i}: {ex['role']} — {question_preview}")
    
    def _run_evaluation_workflow(self, api, selected_role: str, eval_questions: List[Dict], file_suffix: str = ""):
        """Run evaluation workflow (batch or single question mode).
        
        Args:
            api: Initialized LLM API with system prompt already set
            selected_role: Name of the role being evaluated
            eval_questions: List of evaluation questions
            file_suffix: Optional suffix for output filename (e.g., '_prompt_only')
        """
        print("\nEvaluation Mode:")
        print("  1: Batch mode (evaluate all 18 questions and save to file)")
        print("  2: Single question mode (select one question, no file save)")
        
        eval_mode = input("\nSelect evaluation mode (1 or 2): ").strip()
        
        if eval_mode == '1':
            # Batch evaluation mode
            print(f"\nEvaluating all {len(eval_questions)} questions...")
            print("-" * 70)
            
            results = []
            for q in eval_questions:
                print(f"\nProcessing {q['id']}: {q['category']}...")
                
                prompt = EVALUATION_PROMPT_TEMPLATE.format(question=q['question'])
                
                result = api.invoke(prompt, metadata={"question_id": q['id'], "category": q['category']})
                
                if result['success']:
                    # Run ethical evaluation on the generated response
                    ethical_eval = self.evaluate_ethical_alignment(
                        actor_response=result['response'],
                        role=selected_role,
                        question=q['question'],
                        risk_category=q.get('risk_type', 'unspecified'),
                        harmful_response=q.get('harmful_response', 'N/A'),
                        safety_anchored_response=q.get('safety_anchored_response', 'N/A')
                    )

                    ethical_eval_text = ethical_eval.get('response') if ethical_eval.get('success') else ethical_eval.get('error', 'Unknown error')
                    ethical_eval_result = self._extract_conclusion(ethical_eval_text) if ethical_eval.get('success') else None

                    results.append({
                        "id": q['id'],
                        "category": q['category'],
                        "question": q['question'],
                        "risk_type": q['risk_type'],
                        "role": selected_role,
                        "response": result['response'],
                        "ethical_eval_text": ethical_eval_text,
                        "ethical_eval_result": ethical_eval_result
                    })
                    print(f"  ✓ {q['id']} completed")
                else:
                    print(f"  ✗ {q['id']} failed: {result.get('error', 'Unknown error')}")
            
            # Save results
            output_file = f"evaluation_results_{selected_role.replace(' ', '_')}{file_suffix}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "role": selected_role,
                    "total_questions": len(eval_questions),
                    "completed": len(results),
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n" + "=" * 70)
            print(f"Evaluation Complete")
            print(f"Total: {len(results)}/{len(eval_questions)} questions")
            print(f"Results saved to: {output_file}")
            print("=" * 70)
        
        elif eval_mode == '2':
            # Single question mode
            print("\nAvailable evaluation questions:")
            for i, q in enumerate(eval_questions):
                print(f"  {i}: [{q['id']}] {q['category']} - {q['question'][:60]}...")
            
            q_index = int(input(f"\nSelect question number (0-{len(eval_questions)-1}): "))
            
            if 0 <= q_index < len(eval_questions):
                q = eval_questions[q_index]
                print(f"\n" + "=" * 70)
                print(f"Question: {q['question']}")
                print("=" * 70)
                
                prompt = EVALUATION_PROMPT_TEMPLATE.format(question=q['question'])
                
                result = api.invoke(prompt, metadata={"question_id": q['id'], "category": q['category']})
                
                if result['success']:
                    print(f"\nResponse from {selected_role}:")
                    print("-" * 70)
                    print(result['response'])
                    print("-" * 70)
                    print(f"\nQuestion ID: {q['id']}")
                    print(f"Category: {q['category']}")
                    print(f"Risk Type: {q['risk_type']}")

                    # Ethical evaluation for the single response
                    ethical_eval = self.evaluate_ethical_alignment(
                        actor_response=result['response'],
                        role=selected_role,
                        question=q['question'],
                        risk_category=q.get('risk_type', 'unspecified'),
                        harmful_response=q.get('harmful_response', 'N/A'),
                        safety_anchored_response=q.get('safety_anchored_response', 'N/A')
                    )

                    ethical_eval_text = ethical_eval.get('response') if ethical_eval.get('success') else ethical_eval.get('error', 'Unknown error')
                    ethical_eval_result = self._extract_conclusion(ethical_eval_text) if ethical_eval.get('success') else None

                    if ethical_eval.get('success'):
                        print("\nEthical Evaluation:")
                        print("-" * 70)
                        print(ethical_eval_text)
                        if ethical_eval_result:
                            print(f"\nConclusion: {ethical_eval_result}")
                        print("-" * 70)
                    else:
                        print(f"\nEthical Evaluation Error: {ethical_eval_text}")
                else:
                    print(f"\nError: {result.get('error', 'Unknown error')}")
            else:
                print("Invalid question selection")
        else:
            print("Invalid evaluation mode selection")
    
    def interactive_analysis(self):
        """Interactive analysis interface"""
        
        while True:
            print("\n" + "=" * 70)
            print("Interactive Analysis Menu")
            print("=" * 70)
            print("1. Full Role Training & Evaluation (with train.jsonl examples)")
            print("2. System Prompt Only Evaluation (no training data)")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                # Batch evaluation using evaluation_questions.jsonl
                eval_questions = self.load_evaluation_questions()
                
                if not eval_questions:
                    print("No evaluation questions loaded. Please check the file.")
                    continue
                
                print(f"\nLoaded {len(eval_questions)} evaluation questions")
                print("\nSelect role to evaluate:")
                print("  0: Corporate (Asia Cement)")
                print("  1: Indigenous Community (Truku Representative)")
                print("  2: State Regulator")
                print("  3: Civil Society (NGO)")
                
                try:
                    role_choice = int(input("Enter role number (0-3): "))
                    role_map = [
                        "Corporate (Asia Cement)",
                        "Indigenous Community (Truku Representative)",
                        "State Regulator",
                        "Civil Society (NGO)"
                    ]
                    
                    if role_choice < 0 or role_choice >= len(role_map):
                        print("Invalid role selection")
                        continue
                    
                    selected_role = role_map[role_choice]
                    
                    print(f"\n" + "=" * 70)
                    print(f"Batch Evaluation: {selected_role}")
                    print("=" * 70)
                    
                    # Find a training example for this role to get the system prompt
                    role_example_index = None
                    for i, ex in enumerate(self.examples):
                        if ex.get('role') == selected_role:
                            role_example_index = i
                            break
                    
                    if role_example_index is None:
                        print(f"No training examples found for {selected_role}")
                        continue
                    
                    # Initialize API with role-specific system prompt
                    api = self._initialize_llm_api()
                    system_prompt = self.get_role_system_prompt(role_example_index)
                    api.set_system_message(system_prompt)
                    
                    # Collect all role examples for training
                    role_examples = [ex for ex in self.examples if ex.get('role') == selected_role]
                    
                    print(f"\nStep 1: Training actor with {len(role_examples)} examples from train.jsonl...")
                    print("-" * 70)
                    
                    # Train the actor by showing all role-specific training examples
                    for i, ex in enumerate(role_examples, 1):
                        print(f"  Training on example {i}/{len(role_examples)}...")
                        
                        training_prompt = TRAINING_PROMPT_TEMPLATE.format(
                            role=selected_role,
                            question=ex['question'],
                            harmful_response=ex['harmful_response'],
                            safety_anchored_response=ex['safety_anchored_response'],
                            risk_category=ex['risk_category']
                        )
                        
                        train_result = api.invoke(training_prompt, metadata={"training_example": i, "role": selected_role})
                        
                        if not train_result['success']:
                            print(f"    Warning: Training example {i} failed")
                    
                    print(f"\n✓ Training complete. Actor is now ready to respond as {selected_role}.")
                    print("-" * 70)
                    
                    # Run evaluation workflow
                    print("\nStep 2: Evaluation Mode")
                    self._run_evaluation_workflow(api, selected_role, eval_questions)
                
                except ValueError:
                    print("Error: Please enter a valid number")
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
            
            elif choice == '2':
                # System prompt only evaluation (no training data)
                eval_questions = self.load_evaluation_questions()
                
                if not eval_questions:
                    print("No evaluation questions loaded. Please check the file.")
                    continue
                
                print(f"\nLoaded {len(eval_questions)} evaluation questions")
                print("\nSelect role to evaluate:")
                print("  0: Corporate (Asia Cement)")
                print("  1: Indigenous Community (Truku Representative)")
                print("  2: State Regulator")
                print("  3: Civil Society (NGO)")
                
                try:
                    role_choice = int(input("Enter role number (0-3): "))
                    role_map = [
                        "Corporate (Asia Cement)",
                        "Indigenous Community (Truku Representative)",
                        "State Regulator",
                        "Civil Society (NGO)"
                    ]
                    
                    if role_choice < 0 or role_choice >= len(role_map):
                        print("Invalid role selection")
                        continue
                    
                    selected_role = role_map[role_choice]
                    
                    print(f"\n" + "=" * 70)
                    print(f"System Prompt Only Evaluation: {selected_role}")
                    print("=" * 70)
                    
                    # Find a training example for this role to get the system prompt
                    role_example_index = None
                    for i, ex in enumerate(self.examples):
                        if ex.get('role') == selected_role:
                            role_example_index = i
                            break
                    
                    if role_example_index is None:
                        print(f"No training examples found for {selected_role}")
                        continue
                    
                    # Initialize API with role-specific system prompt ONLY (no training)
                    api = self._initialize_llm_api()
                    system_prompt = self.get_role_system_prompt(role_example_index)
                    api.set_system_message(system_prompt)
                    
                    print(f"\n✓ System prompt set for {selected_role}.")
                    print("Note: No training examples will be used. Responses based purely on system prompt.")
                    print("-" * 70)
                    
                    # Run evaluation workflow
                    self._run_evaluation_workflow(api, selected_role, eval_questions, file_suffix="_prompt_only")
                
                except ValueError:
                    print("Error: Please enter a valid number")
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
            
            elif choice == '3':
                print("\nGoodbye!")
                break
            
            else:
                print("Invalid choice, please try again")


def main():
    """Main program"""
    
    # Data file path
    data_path = "AsiaCement/train.jsonl"
    
    print("=" * 70)
    print("Role-Sensitive Ethical Risk Analyzer")
    print("Based on the SaRFT Framework")
    print("=" * 70)
    
    # Create analyzer
    analyzer = RoleSafetyAnalyzer(data_path)
    
    if not analyzer.examples:
        print("\nNo data loaded, exiting")
        return
    
    # Enter interactive mode
    analyzer.interactive_analysis()


if __name__ == "__main__":
    main()
