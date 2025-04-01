# Model Evolution Engine: A Step Toward Self-Evolving AI

The **Model Evolution Engine** is a self-improving artificial intelligence (AI) system designed for iterative code generation and optimization. It integrates **DeepSeek** for enhancement suggestions, **CodeLlama** for code refinement, and an **NMT model with Bahdanau Attention** as the foundation. By leveraging **AST-based metadata extraction** and a **FAISS-driven retrieval mechanism**, the system improves NMT pipeline efficiency, making it suitable for resource-constrained environments.

This project demonstrates a scalable framework for automated software development and contributes to the advancement of self-evolving AI applications.

## Features

**Self-Improving AI Framework**: Iteratively enhances code through a multi-model architecture.

**DeepSeek Integration**: Analyzes NMT pipeline code to suggest optimizations.

**CodeLlama Refinement**: Converts DeepSeek’s suggestions into executable improvements.

**NMT Model with Bahdanau Attention**: Serves as the core model, implemented in PyTorch.

**FAISS-Driven Optimization**: Retains high-quality improvements using a scoring mechanism.

**Performance Metrics Tracking**:

- Execution Time Reduction: 12.47% faster preprocessing (11.36s → 9.94s).

- Model Execution Speedup: 15.44% faster (7.12s → 6.02s).

- Complexity and Maintainability Trade-offs: Slight increase in complexity (16 → 19) and reduction in maintainability (166 → 144).

**Hardware Efficiency**: Runs on 8GB RAM with an NVIDIA GTX 1650 (4GB VRAM).

**Ollama for Model Management**: Simplifies model downloads and execution.

## Prerequisites

- Python 3.10

## Engine Architecture

The engine consists of three core models: **DeepSeek**, **CodeLlama**, and a **Neural Machine Translation (NMT)** model. These models work in tandem to enhance the quality, efficiency, and performance of the NMT pipeline.

### 1. Overview of Engine Architecture

The system is structured as follows:

- **DeepSeek Model**: Provides structured improvement suggestions for the NMT pipeline.

- **CodeLlama Model**: Synthesizes and integrates improvements into the NMT pipeline.

- **Neural Machine Translation (NMT) Model**: Performs English-to-French translation using an encoder-decoder architecture.

- **FAISS Vector Database**: Stores high-quality improvement suggestions and generated code for future reference.

- **Metadata and AST Analysis**: Extracts structural information to enhance code generation accuracy.

### 2. DeepSeek Model

The DeepSeek model is responsible for analyzing the NMT pipeline and generating structured improvement recommendations. The system utilizes the DeepSeek-R1 variant, which assesses the quality, structure, and maintainability of the pipeline code. Its primary functions include:

- Identifying inefficiencies and suggesting optimization strategies.

- Recommending modifications to improve maintainability and readability.

- Providing structured suggestions that guide refinement of the pipeline.

### 3. CodeLlama Model

The CodeLlama model translates the suggestions from DeepSeek into executable code improvements. The CodeLlama 7B variant is used, ensuring structured feedback integration. Key functionalities of CodeLlama include:

- Generating optimized code based on DeepSeek’s suggestions.

- Enhancing the quality, maintainability, and efficiency of the NMT pipeline.

- Utilizing metadata and Abstract Syntax Tree (AST) analysis for context-aware code synthesis.

### 4. Neural Machine Translation (NMT) Model

The NMT model functions as a sequence-to-sequence translation system, leveraging the Bahdanau Attention mechanism. The architecture follows an encoder-decoder structure for English-to-French translation, utilizing:

- Marian Tokenizer from the Transformer library (Helsinki-NLP/opus-mt-en-fr) for tokenization.

- Key configurations:

  - Embedding dimension: 128

  - Hidden dimension: 512

  - Number of LSTM layers: 4

  - Dropout: 0.4

### 5. FAISS Vector Database

The engine incorporates **FAISS (Facebook AI Similarity Search)** to store high-quality suggestions and generated code that meet a predefined quality threshold (≥25/30). This database facilitates:

- Efficient retrieval of past successful modifications to guide future improvements.

- Retrieval-Augmented Generation (RAG) by providing high-quality examples as context for subsequent AI-generated refinements.

- Continuous enhancement of the NMT pipeline through iterative retrieval and refinement based on past successful outputs.

### 6. Metadata and Abstract Syntax Tree (AST) Analysis

The engine maintains a metadata file (metadata.json) to store structural information about the NMT pipeline. This is achieved through AST (Abstract Syntax Tree) analysis, which extracts:

- Function and class signatures.

- Structural dependencies within the code.

- Contextual metadata that enhances CodeLlama’s ability to generate relevant modifications.

### 7. Workflow of the Engine

**1. DeepSeek Analysis**: The DeepSeek model analyzes the NMT pipeline and generates structured suggestions.

**2. CodeLlama Refinement**: CodeLlama integrates these suggestions into executable code modifications.

**3. Metadata Utilization**: AST analysis and metadata extraction improve CodeLlama’s understanding of the pipeline structure.

**4. Quality Evaluation**: The modified code is evaluated, and high-quality outputs are stored in the FAISS database.

**5. Iterative Enhancement**: The system continuously references stored improvements for future refinements.

This architecture ensures an **adaptive, self-improving translation engine** that integrates **feedback-driven optimization** through **Retrieval-Augmented Generation (RAG)** while leveraging **iterative refinement based on past successful modifications** to enhance performance over time.

## Experimental Results

Due to hardware constraints, only two key files—`preprocessing.py` and `model.py`—were enhanced within the Neural Machine Translation (NMT) pipeline.

**1. Enhancements in `preprocessing.py`**

- **Functionality**: Handles data loading, tokenization, dataset splitting, and tokenizer persistence for NMT initialization.

- **DeepSeek’s Suggestions**: Generated recommendations to improve data loading efficiency.

- **Modifications by CodeLlama**:

  - Optimized the `load_data` function by introducing chunk-based processing to reduce memory overhead.

- **Performance Improvements**:

  - Execution Speed: Increased by 12.47%.

  - Cyclomatic Complexity: Increased from 16 to 19, indicating additional decision-making operations.

  - Maintainability Index: Dropped from 166 to 144, suggesting that while efficiency improved, code complexity increased.

**2. Enhancements in `model.py`**

- **Functionality**: Implements a sequence-to-sequence NMT model.

- **DeepSeek’s Suggestions**: Recommended replacing Bahdanau Attention with Scaled Dot-Product Attention for better performance.

- **Modifications by CodeLlama**:

  - Successfully implemented the Scaled Dot-Product Attention mechanism.

- **Performance Improvements**:

  - Execution Speed: Improved by 15.44%.

  - Cyclomatic Complexity: Unchanged at 18, suggesting the structural complexity remained stable.

  - Maintainability Index: Dropped slightly from 100 to 97, likely due to additional attention mechanism logic.

### Overall Findings

- The preprocessing stage benefited from chunk-based data loading, improving speed but increasing code complexity.

- The model architecture update (switching attention mechanisms) significantly boosted performance while maintaining structural simplicity.

- Despite a slight decline in maintainability, the improvements in execution speed demonstrate the effectiveness of these enhancements.

## Evaluation Process for AI-Generated Code in the NMT Pipeline

After AI-generated code is produced, it undergoes a structured three-stage evaluation process to assess improvements in complexity, maintainability, execution, and compatibility.

### 1. Three-Stage Evaluation Process

**Stage 1: Code Analysis**

- **Objective**: Evaluate the structural complexity and maintainability of the AI-generated code.

- **Metrics Computed**:

  - Cyclomatic Complexity: Measures the number of independent execution paths in the code.

  - Maintainability Index: Assesses code readability and ease of modification.

- **Reward Mechanism**:

  - If the AI-generated code shows better performance in either metric, the reward increases by 10 points (5 points for each metric).

**Stage 2: Code Execution**

- **Objective**: Ensure the AI-generated code runs successfully and improves performance.

- **Process**:

  1. The system creates a backup of the original file.

  2. The AI-generated code replaces the original file.

  3. Both the original and AI-generated versions execute in a shell environment to compare performance.

  4. If the execution completes without errors and meets predefined performance criteria, the reward increases by 10 points.

  5. The original file is restored from the backup, and the backup is deleted.

**Stage 3: DVC Pipeline Execution**

- **Objective**: Validate compatibility of AI-generated code within the full NMT pipeline.

- **Process**:

  1. A second backup of the original file is created.

  2. The AI-generated code replaces the original file.

  3. The entire NMT pipeline is executed using the command:

        ```sh
        dvc repro --force
        ```

        ensuring that no dependencies are broken.

  4. If the pipeline executes successfully, the reward increases by 10 points.

  5. The original file is restored, and the backup is deleted.

### 2. Evaluation Outputs

Upon completing all three stages, the system records the following key results:

- **Evaluation Status**: Boolean value (`True` if successful, `False` if failed).

- **Reward Score**: Cumulative score based on performance improvements.

- **Error Message**: If applicable, the error encountered during execution.

### 3. Post-Evaluation Actions

**Case 1: Evaluation Status = `True` (Success)**

- The AI-generated code permanently replaces the original code in the NMT pipeline.

- Metadata is updated to reflect the changes.

- If the reward score exceeds the acceptance threshold, the **FAISS** vector database is updated with:

  - DeepSeek model's improvement suggestions

  - CodeLlama’s modified code

**Case 2: Evaluation Status = `False` (Failure)**

- The engine modifies the prompt, appending a new section that includes error details.

- The revised prompt is fed back into the CodeLlama model, ensuring the same error does not occur.

- Each file in the NMT pipeline is given up to five attempts for successful enhancement.

- If all five attempts fail, the file is skipped for manual review or alternative refinement strategies.

### 4. Impact of the Evaluation Process

- Iterative Refinement: AI-generated code undergoes multiple rounds of improvements based on performance feedback.

- Automated Debugging: Errors encountered during execution are used to enhance the AI’s future generations.

- Optimized NMT Pipeline: Ensures improved maintainability, execution speed, and structural efficiency.

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/Model-Evolution-Engine.git
cd Model-Evolution-Engine
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initialize DVC Pipeline

```sh
dvc init
```

### Run the Engine

```sh
python main.py
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.