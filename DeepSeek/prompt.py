def generate_suggestion_prompt(code: str, metadata: str):
    return [
        {
            "role": "user",
            "content": f"""You are an expert deep learning engineer.  
            Analyze the given code and provide exactly **one** high-impact improvement suggestion.  

            **Guidelines:**  
            - Focus on **Model performance, memory efficiency, and execution speed**.  
            - **Do NOT modify data formats** (e.g., CSV, JSON, or NPZ) to ensure pipeline compatibility.  
            - Ensure suggestions **preserve existing data structures and I/O formats**.  
            - Be specific—mention exact **functions, libraries, or techniques**.  
            - Use metadata to ensure project consistency.  
            - Respond **only** with the suggestion in **text markdown** format (no explanations).  

            **Code:**  
            ```python  
            {code}  
            ```  

            **Project Metadata:**  
            {metadata}  

            **Expected Output Format:**  
            **Suggestion:** <Your improvement suggestion here>  
            """,
        }
    ]
