from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from config import config
import json

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = config.EMBEDDING_DIMENSION
        self.medical_knowledge = self._load_medical_knowledge()
    
    def _load_medical_knowledge(self) -> List[Dict[str, Any]]:
        """Load medical knowledge base for RAG"""
        # This is a simplified knowledge base. In production, you'd load from your database
        knowledge_base = [
            {
                "disease": "hypertension",
                "questions": [
                    "When did you first notice your blood pressure was elevated?",
                    "Do you monitor your blood pressure at home? What are typical readings?",
                    "Do you experience headaches, particularly in the morning?",
                    "Have you noticed any dizziness or lightheadedness?",
                    "Do you have a family history of high blood pressure?",
                    "What is your typical salt intake?",
                    "How often do you exercise?",
                    "Do you experience any chest pain or shortness of breath?",
                    "Are you currently taking any medications for blood pressure?"
                ]
            },
            {
                "disease": "diabetes",
                "questions": [
                    "When were you first diagnosed with diabetes?",
                    "What is your most recent HbA1c level?",
                    "Do you monitor your blood sugar at home? What are typical readings?",
                    "Have you experienced increased thirst or urination?",
                    "Do you experience blurred vision?",
                    "Have you noticed any unexplained weight loss or gain?",
                    "Do you experience frequent infections or slow healing wounds?",
                    "Do you have tingling or numbness in your hands or feet?",
                    "What is your typical diet like?",
                    "What medications are you currently taking for diabetes?"
                ]
            },
            {
                "disease": "asthma",
                "questions": [
                    "When did you first experience asthma symptoms?",
                    "How often do you experience wheezing or shortness of breath?",
                    "What triggers your asthma symptoms?",
                    "Do you wake up at night due to breathing difficulties?",
                    "How often do you use your rescue inhaler?",
                    "Do you experience chest tightness?",
                    "Do you have a cough, especially at night or early morning?",
                    "Have you been hospitalized for asthma?",
                    "Do you have any allergies?",
                    "Does exercise make your symptoms worse?"
                ]
            },
            {
                "disease": "migraine",
                "questions": [
                    "How long have you been experiencing migraines?",
                    "How often do you get migraines?",
                    "How long does each migraine typically last?",
                    "Where is the pain located?",
                    "On a scale of 1-10, how would you rate the pain?",
                    "Do you experience any warning signs before a migraine?",
                    "Do you have sensitivity to light or sound during migraines?",
                    "Do you experience nausea or vomiting?",
                    "What triggers your migraines?",
                    "What helps relieve your migraine pain?"
                ]
            },
            {
                "disease": "depression",
                "questions": [
                    "How long have you been feeling this way?",
                    "Have you noticed changes in your sleep patterns?",
                    "Have you experienced changes in appetite or weight?",
                    "Do you find it difficult to concentrate or make decisions?",
                    "Have you lost interest in activities you used to enjoy?",
                    "Do you feel tired or have low energy most days?",
                    "Have you had thoughts of harming yourself?",
                    "Do you feel hopeless or worthless?",
                    "Have you experienced any major life changes recently?",
                    "Do you have a support system of family or friends?"
                ]
            },
            {
                "disease": "arthritis",
                "questions": [
                    "Which joints are affected?",
                    "When did you first notice joint pain or stiffness?",
                    "Is the pain worse in the morning or evening?",
                    "How long does morning stiffness last?",
                    "Does the pain improve with movement or rest?",
                    "Have you noticed any swelling in your joints?",
                    "Do you experience warmth or redness in affected joints?",
                    "How does the pain affect your daily activities?",
                    "Have you noticed any joint deformity?",
                    "Do you have a family history of arthritis?"
                ]
            },
            {
                "disease": "gerd",
                "questions": [
                    "How often do you experience heartburn?",
                    "When do symptoms typically occur (after meals, at night)?",
                    "Do you experience a sour or bitter taste in your mouth?",
                    "Do you have difficulty swallowing?",
                    "Do you experience chest pain?",
                    "Do you have a chronic cough or hoarseness?",
                    "What foods seem to trigger your symptoms?",
                    "Do symptoms worsen when lying down?",
                    "Have you noticed any regurgitation of food or liquid?",
                    "What medications have you tried for relief?"
                ]
            },
            {
                "disease": "anxiety",
                "questions": [
                    "How long have you been experiencing anxiety?",
                    "What situations trigger your anxiety?",
                    "Do you experience panic attacks?",
                    "Do you have physical symptoms like rapid heartbeat or sweating?",
                    "Do you have difficulty sleeping due to worry?",
                    "Do you find it hard to control your worrying?",
                    "Do you avoid certain situations due to anxiety?",
                    "Do you experience muscle tension or restlessness?",
                    "How does anxiety affect your daily life?",
                    "Have you tried any coping strategies?"
                ]
            }
        ]
        
        # Create embeddings for each question
        for disease_info in knowledge_base:
            questions = disease_info["questions"]
            embeddings = self.embedding_model.encode(questions)
            disease_info["embeddings"] = embeddings
        
        return knowledge_base
    
    def get_relevant_questions(self, ailment: str, current_context: str, asked_questions: List[str], top_k: int = 3) -> List[str]:
        """Retrieve relevant follow-up questions based on context"""
        # Find the disease knowledge base
        disease_kb = None
        ailment_lower = ailment.lower()
        
        for kb in self.medical_knowledge:
            if kb["disease"].lower() in ailment_lower or ailment_lower in kb["disease"].lower():
                disease_kb = kb
                break
        
        if not disease_kb:
            # Generic questions if disease not found
            return [
                "Can you describe your main symptoms?",
                "When did these symptoms first start?",
                "How severe are your symptoms on a scale of 1-10?"
            ]
        
        # Encode current context
        context_embedding = self.embedding_model.encode([current_context])[0]
        
        # Calculate similarity scores
        similarities = np.dot(disease_kb["embeddings"], context_embedding)
        
        # Get top questions that haven't been asked
        sorted_indices = np.argsort(similarities)[::-1]
        
        relevant_questions = []
        for idx in sorted_indices:
            question = disease_kb["questions"][idx]
            if question not in asked_questions and len(relevant_questions) < top_k:
                relevant_questions.append(question)
        
        return relevant_questions
    
    def extract_information_categories(self, ailment: str) -> List[str]:
        """Get relevant information categories for an ailment"""
        categories = {
            "hypertension": ["onset", "readings", "symptoms", "family_history", "lifestyle", "medications"],
            "diabetes": ["diagnosis_date", "hba1c", "blood_sugar", "symptoms", "complications", "diet", "medications"],
            "asthma": ["onset", "frequency", "triggers", "severity", "medications", "allergies"],
            "migraine": ["duration", "frequency", "pain_characteristics", "aura", "triggers", "relief_methods"],
            "depression": ["duration", "sleep", "appetite", "concentration", "interests", "energy", "thoughts"],
            "arthritis": ["affected_joints", "onset", "pain_pattern", "stiffness", "swelling", "impact"],
            "gerd": ["frequency", "timing", "symptoms", "triggers", "severity", "medications"],
            "anxiety": ["duration", "triggers", "panic_attacks", "physical_symptoms", "impact", "coping"]
        }
        
        ailment_lower = ailment.lower()
        for disease, cats in categories.items():
            if disease in ailment_lower or ailment_lower in disease:
                return cats
        
        return ["chief_complaint", "onset", "duration", "severity", "associated_symptoms"]

vector_store = VectorStore()