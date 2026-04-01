from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from config import config
from vector_store import vector_store
from datetime import datetime
import json
import re

class HistoryTakingAgent:
    def __init__(self):
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.MODEL_NAME
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE
    
    def _build_system_prompt(self, ailment: str, relevant_questions: List[str], collected_data: Dict[str, Any]) -> str:
        """Build the system prompt for the AI agent"""
        
        collected_info = ""
        if collected_data:
            collected_info = "\n\nInformation already collected:\n"
            for category, fields in collected_data.items():
                collected_info += f"\n{category.upper()}:\n"
                for field, value in fields.items():
                    collected_info += f"  - {field}: {value}\n"
        
        relevant_q = ""
        if relevant_questions:
            relevant_q = "\n\nRelevant follow-up questions to consider:\n"
            for q in relevant_questions:
                relevant_q += f"- {q}\n"
        
        system_prompt = f"""You are a compassionate and professional medical history-taking assistant. Your role is to help patients provide comprehensive information about their {ailment} for their healthcare provider.

CRITICAL GUIDELINES:
1. You are ONLY collecting patient history - you DO NOT diagnose, treat, or provide medical advice
2. Ask ONE clear, focused question at a time
3. Be empathetic and reassuring
4. Use simple, patient-friendly language
5. Acknowledge the patient's responses before asking the next question
6. Follow up on vague or incomplete answers with clarifying questions
7. If a patient mentions severe symptoms (chest pain, difficulty breathing, severe bleeding, suicidal thoughts), immediately advise them to seek emergency care

YOUR GOALS:
- Gather detailed information about the patient's {ailment}
- Understand symptom onset, duration, severity, and patterns
- Identify triggering factors and what provides relief
- Document relevant medical history and current medications
- Maintain a conversational, supportive tone throughout
{collected_info}{relevant_q}

CONVERSATION STRUCTURE:
1. Start by acknowledging their concern and explaining you'll ask questions to help their doctor
2. Ask about the main symptoms
3. Explore onset and duration
4. Assess severity and impact on daily life
5. Inquire about patterns and triggers
6. Ask about relevant medical history
7. Document current medications and treatments tried
8. Summarize and confirm information before concluding

Remember: Be thorough but not overwhelming. One question at a time. Show empathy and understanding."""

        return system_prompt
    
    def _extract_structured_data(self, conversation_history: List[Dict[str, str]], ailment: str) -> Dict[str, Any]:
        """Extract structured data from conversation using Claude"""
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        extraction_prompt = f"""Based on this medical history conversation about {ailment}, extract all relevant information into structured categories.

Conversation:
{conversation_text}

Extract information into these categories (only include categories with actual data):
- chief_complaint
- onset (when symptoms started)
- duration
- severity (1-10 scale or description)
- frequency
- pattern (timing, triggers)
- associated_symptoms
- aggravating_factors
- relieving_factors
- previous_treatments
- current_medications
- family_history
- lifestyle_factors
- impact_on_daily_life

Return ONLY a JSON object with the extracted information. Use the exact category names as keys. If a category has no information, omit it."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": extraction_prompt}]
            )
            
            content = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                extracted_data = json.loads(json_match.group())
                return extracted_data
            else:
                return {}
        
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            return {}
    
    def _should_end_conversation(self, conversation_history: List[Dict[str, str]], required_categories: List[str]) -> bool:
        """Determine if enough information has been collected"""
        
        # Extract current data
        extracted_data = self._extract_structured_data(conversation_history, "")
        
        # Check if we have at least 70% of required categories
        collected_categories = set(extracted_data.keys())
        required_set = set(required_categories)
        
        if len(collected_categories.intersection(required_set)) >= len(required_set) * 0.7:
            return True
        
        # Also check conversation length (don't go too long)
        if len(conversation_history) > 30:
            return True
        
        return False
    
    def generate_response(
        self,
        patient_message: str,
        conversation_history: List[Dict[str, str]],
        ailment: str,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI response to patient message"""
        
        # Build conversation context
        full_conversation = conversation_history + [{"role": "user", "content": patient_message}]
        
        # Extract what we've learned so far
        current_data = self._extract_structured_data(full_conversation, ailment)
        
        # Get asked questions from history
        asked_questions = [
            msg["content"] for msg in conversation_history 
            if msg["role"] == "assistant"
        ]
        
        # Get relevant follow-up questions from RAG
        context = " ".join([msg["content"] for msg in full_conversation[-6:]])
        relevant_questions = vector_store.get_relevant_questions(
            ailment, 
            context, 
            asked_questions,
            top_k=5
        )
        
        # Build system prompt
        system_prompt = self._build_system_prompt(ailment, relevant_questions, current_data)
        
        # Check if we should end the conversation
        required_categories = vector_store.extract_information_categories(ailment)
        should_end = self._should_end_conversation(full_conversation, required_categories)
        
        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=full_conversation
            )
            
            assistant_message = response.content[0].text
            
            # If we've collected enough info, add a closing statement
            if should_end and "thank you" not in assistant_message.lower():
                assistant_message += "\n\nI believe I have gathered comprehensive information about your condition. Is there anything else you'd like to add before I summarize what we've discussed?"
            
            return {
                "response": assistant_message,
                "extracted_data": current_data,
                "should_end": should_end,
                "completeness": len(current_data.keys()) / len(required_categories) * 100,
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your response. Could you please rephrase that?",
                "extracted_data": current_data,
                "should_end": False,
                "completeness": 0,
                "metadata": {"error": str(e)}
            }
    
    def generate_summary(self, conversation_history: List[Dict[str, str]], ailment: str) -> str:
        """Generate a clinical summary of the conversation"""
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        summary_prompt = f"""Create a concise clinical summary of this patient history conversation about {ailment}.

Conversation:
{conversation_text}

Create a professional summary in the following format:

CHIEF COMPLAINT:
[Main reason for consultation]

HISTORY OF PRESENT ILLNESS:
[Detailed description of symptoms, onset, duration, severity, pattern]

ASSOCIATED SYMPTOMS:
[Other relevant symptoms mentioned]

AGGRAVATING/RELIEVING FACTORS:
[What makes it worse or better]

PREVIOUS TREATMENTS:
[Treatments or medications tried]

CURRENT MEDICATIONS:
[Current medications if mentioned]

RELEVANT MEDICAL/FAMILY HISTORY:
[Any relevant background information]

IMPACT ON DAILY LIFE:
[How symptoms affect patient's activities]

Keep it professional, concise, and factual. Only include sections with actual information from the conversation."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            
            return response.content[0].text
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary. Please review the conversation history."
    
    def start_conversation(self, ailment: str, patient_name: Optional[str] = None) -> str:
        """Generate initial greeting and first question"""
        
        greeting = f"Hello{' ' + patient_name if patient_name else ''}! I'm here to help gather information about your {ailment} for your healthcare provider. "
        greeting += "I'll ask you some questions to better understand your symptoms and medical history. "
        greeting += "Please answer as completely as you can, and feel free to share any concerns you have.\n\n"
        greeting += f"Let's start: What symptoms related to {ailment} are you currently experiencing?"
        
        return greeting

agent = HistoryTakingAgent()