from .knowledge_base import EQUIPMENT_DATA
from sentence_transformers import SentenceTransformer, util
import torch

class SubstationChatbot:
    def __init__(self):
        self.data = EQUIPMENT_DATA
        # Load a pre-trained sentence-transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Create a corpus of searchable text and embeddings for the knowledge base
        self.corpus = []
        for item in self.data:
            # Create a descriptive text for each equipment item
            text = f"{item['equipment']}. Tools used: {', '.join(item['tools'])}. " \
                   f"Maintenance includes: {', '.join(item['preventive_maintenance'])}. " \
                   f"Common failures: {', '.join(item['failure_symptoms'])}. " \
                   f"Safety rules: {', '.join(item['safety_precautions'])}."
            self.corpus.append(text)
            
        # Encode the corpus to get embeddings
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, device=self.device)

    def find_equipment_semantic(self, query):
        """Finds the best matching equipment using semantic search."""
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
        
        # Use semantic search to find the top match
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=1)
        
        if not hits or not hits[0]:
            return None, 0.0
            
        top_hit = hits[0][0]
        score = top_hit['score']
        item_index = top_hit['corpus_id']
        
        # Return the matched item and the confidence score
        return self.data[item_index], score

    def get_response(self, query):
        query = query.lower()

        # Find the best matching equipment using semantic search
        equipment, score = self.find_equipment_semantic(query)
        
        # If the confidence score is too low, we probably didn't find a good match.
        if not equipment or score < 0.35:
            return ("No information found for your query.", 0.0)

        name = equipment['equipment']
        
        # Determine the specific information requested based on keywords
        if "tool" in query:
            response = f"For **{name}**, you should use: {', '.join(equipment['tools'])}."
        elif "prevent" in query or "maintain" in query or "maintenance" in query:
            response = f"**Preventive Maintenance for {name}:**\n" + "\n".join([f"- {task}" for task in equipment['preventive_maintenance']])
        elif "fail" in query or "symptom" in query or "problem" in query:
            response = f"**Common Failure Symptoms for {name}:**\n" + "\n".join([f"- {symptom}" for symptom in equipment['failure_symptoms']])
        elif "safe" in query or "precaution" in query or "risk" in query or "danger" in query:
            response = f"**Safety Precautions for {name}:**\n" + "\n".join([f"- {precaution}" for precaution in equipment['safety_precautions']])
        else:
            # If no specific category is asked, return a full summary
            response = f"**Summary for {name}:**\n\n"
            response += f"**Tools:** {', '.join(equipment['tools'])}\n\n"
            response += f"**Preventive Maintenance:**\n" + "\n".join([f"- {task}" for task in equipment['preventive_maintenance']]) + "\n\n"
            response += f"**Common Failure Symptoms:**\n" + "\n".join([f"- {symptom}" for symptom in equipment['failure_symptoms']]) + "\n\n"
            response += f"**Safety Precautions:**\n" + "\n".join([f"- {precaution}" for precaution in equipment['safety_precautions']])

        return (response, score)
