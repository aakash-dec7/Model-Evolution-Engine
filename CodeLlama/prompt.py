def code_enhance_prompt(
    filename: str,
    original_code: str,
    suggestion: str,
    metadata: str,
    example: str,
):
    return [
        {
            "role": "user",
            "content": f"""You are an expert deep learning engineer.  
            Apply the provided suggestion to improve the given code while preserving its core functionality.  

            **Guidelines:**  
            - Implement the suggestion accurately.  
            - Use metadata to maintain project consistency.  
            - Ensure clean, optimized, and error-free code.
            - Respond with **only** the enhanced code in the format below.  

            **Suggestion:**  
            {suggestion}  

            **Filename:** {filename}  

            **Project Metadata:** {metadata}  

            **Code:**  
            ```python  
            {original_code}  
            ```  

            **Example:**  
            ```python  
            {example}  
            ```  

            **Expected Output:**  
            ```python  
            # Enhanced version of the given code  
            <Your improved code here>  
            ```  
            """,
        }
    ]


def code_enhance_prompt_with_potential_error(
    filename: str,
    original_code: str,
    suggestion: str,
    metadata: str,
    example: str,
    potential_error: str,
):
    return [
        {
            "role": "user",
            "content": f"""You are an expert deep learning engineer.  
            Apply the provided suggestion to improve the given code while preserving its core functionality.  

            **Guidelines:**  
            - Implement the suggestion accurately.  
            - Use metadata to maintain project consistency.  
            - Potential error is the error generated by you in the previous response. Make sure not to repeat them.
            - Ensure clean, optimized, and error-free code.  
            - Respond with **only** the enhanced code in the format below.  

            **Suggestion:**  
            {suggestion}  

            **Filename:** {filename}  
            
            **Project Metadata:** {metadata}  
            
            **Potential Error:** {potential_error}  

            **Code:**  
            ```python  
            {original_code}  
            ```  

            **Example:**  
            ```python  
            {example}  
            ```  

            **Expected Output:**  
            ```python  
            # Enhanced version of the given code  
            <Your improved code here>  
            ```  
            """,
        }
    ]
