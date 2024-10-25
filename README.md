# Character-Level-Transformer-Model-for-Text-Synthesis
A simple foundational PyTorch-based model for character-level language modeling

---

### Title and Details

**Title**: Character-Level-Transformer-Model-for-Text-Synthesis

**Details**: This project implements a character-level language model using a multi-layer Transformer architecture. The model is capable of generating text character-by-character, trained on sequence data with customizable hyperparameters. Key features include training on unique datasets, a configurable Transformer model, and CUDA support for optimized performance.

---

### Dataset Descriptions

This project uses two datasets:

- **Shakespeare Dataset (`shakespeare.txt`)**: This dataset contains a collection of texts attributed to William Shakespeare. It is used to train the model to generate Shakespearean text in a style that reflects the vocabulary and phrasing of Shakespeare's works.
  
- **Bible Dataset (`bible.txt`)**: This dataset contains text from the Bible, providing an alternative training set for the model. It allows the model to generate religious-themed language, mimicking the structure and style commonly found in biblical texts.

These datasets are treated as character-level sequences, where each character is mapped to an integer, enabling the model to learn sequential character dependencies.

---

### Generated Text Files

The model generates two output files after training on each dataset:

1. **`fakespeare.txt`**: Contains the model’s generated text based on training with the Shakespeare dataset (`shakespeare.txt`). The output reflects the model’s learned Shakespearean language style, with character interactions, stylistic dialogue, and vocabulary inspired by classic Shakespearean works.

2. **`bibble.txt`**: Contains the model’s generated text based on training with the Bible dataset (`bible.txt`). The model mimics the style and structure of religious texts, using vocabulary and phrasing characteristic of biblical scripture.

These files are created using the following lines in the script:
```python
# Write the generated Shakespeare-style text to 'fakespeare.txt'
open('fakespeare.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# Write the generated Bible-style text to 'bibble.txt'
open('bibble.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
```

Each file showcases the model’s ability to adapt to different language styles and offers a unique synthesis of text based on the chosen training data. These outputs can serve as examples of character-level generation and help visualize the learned stylistic features in each domain. 

---

### Project Overview

This project builds a character-level language model using PyTorch, with a multi-layer Transformer architecture. The model is trained to predict the next character in a sequence based on prior context, using a self-attention mechanism to learn patterns within text data. The project explores different contexts and text structures by training the model on two distinct datasets: Shakespeare’s works and the Bible. Key objectives include training a robust model for text generation, experimenting with context length, and learning rate optimization.

---

### Installation

To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gummadianand27/Character-Level-Transformer-Model-for-Text-Synthesis.git
   ```

2. **Install required packages**:
   - Make sure you have Python 3.7 or later installed.
   - Install the required libraries:
     ```bash
     pip install torch
     ```

3. **Datasets**:
   - Include `shakespeare.txt` and `bible.txt` in the project root directory. These should contain the Shakespeare and Bible texts, respectively, for model training.

---

### Usage

1. **Configure Hyperparameters**: Adjust parameters like `batch_size`, `block_size`, and `learning_rate` within the `foundation.py` script to suit your training requirements.

2. **Run Training**: Execute the following command to start training:
   ```bash
   python foundation.py
   ```

3. **Monitor Training**: The script provides console output at intervals to monitor the training loss and performance on evaluation data.

4. **Generate Text**: After training, use the model to generate text by providing a starting string and using the model to predict successive characters.

---

### Results

#### Results on `shakespeare.txt` (Shakespeare Dataset)

The model training on `shakespeare.txt` yields promising results with gradual improvement in both training and validation loss. Here's a summary of the key outcomes:

- **Model Complexity**: The Transformer model has approximately **10.79 million parameters**, allowing it to capture complex character-level dependencies.
  
- **Training Progress**:
  - **Early Training**: Starts with a high initial training loss of 4.2221, but quickly improves within the first 500 steps, reaching a training loss of 1.7473 and a validation loss of 1.9011.
  - **Later Stages**: By step 4999, training loss reduces to 0.8551, and validation loss reaches 1.5520, indicating the model has learned the character-level patterns in the Shakespearean text with good generalization.

- **Text Generation**:
    ```
    SABELLA:
    Badness to I cannot love like by him.
    
    Proud I am sorry the rude anchor,
    And to that his life to him. Nay, no doubt
    I would lose him.
    
    POLIXENES:
    Lady, shall none.
    I'll unkull afraid to privily: come,
    When Barnardina, we bid.
    
    LUCIO:
    There was painted by your bed.
    
    LUCIO:
    Good my lords brother through his kindred courtier,
    But if it rever be consumed by all undertakind, 
    Adding you not like speak.
    
    PAULINA:
    Farewell;
    The duke is well:
    Let it be kind off a scene kind's right. 
    ```
  - After training, the model produces text that mimics the structure and style of Shakespeare’s language.
  - Example generated text shows coherent roles with distinct characters (ISABELLA, POLIXENES, LUCIO, PAULINA) and structured dialogue, even with complex sentence formations and dramatic tone.
  - This output demonstrates the model’s capability to replicate character dialogues and generate text that, while not perfect, shows a strong grasp of Shakespearean language structure, vocabulary, and tone.

#### Results on `bible.txt` (Bible Dataset)

The model demonstrates effective learning on `bible.txt`, showing a steady decrease in both training and validation losses over time. Here's a detailed summary:

- **Model Complexity**: With around **10.80 million parameters**, the model is well-suited to capturing the distinctive style and phrasing typical of biblical texts.

- **Training Progress**:
  - **Initial Steps**: The training starts with a high initial loss of 4.3476, indicating the model is beginning from scratch with no understanding of the text’s structure. By step 500, however, it has made significant progress, achieving a training loss of 1.4375 and validation loss of 1.6297.
  - **Final Stages**: At the end of 5000 steps, the model reaches a training loss of 0.7760 and validation loss of 1.1676, suggesting a good level of generalization and strong performance on the dataset.

- **Generated Text**:
```
That we should not have done this thou beguale on high: but I am heard of Saul, and kissed he also munt the voice of Samaria.
  
And he went from Saul, and he was forty years old after king Solomon. And Gazar and Uriel the elders of the conkensidged between Rerahah's house, and brake heared Adah. Now the people was weight forty and drinking war against Ramah, Rebekah, between Gaash; and the brother of Jair thou asked Abner all them. And Lazarus went up: and about his sisster, and missons and hips
```

  - The generated text reflects the biblical language structure, with phrases that resemble religious and historical storytelling ("...the voice of Samaria," "forty years old after king Solomon").
  - Although some sentences are syntactically unclear, the model effectively captures the formal and narrative tone of biblical texts.
  - Examples show the use of biblical names and structures common in scripture, such as "Saul," "Solomon," and "Rebekah," along with sentence constructs that mimic the text’s style.

These results indicate that the model successfully learns the character-level language patterns from both datasets, with clear adaptation to each text’s unique linguistic style and structure.

---

### Contributions

- **Gummadianand27**: Developed and implemented the Transformer-based character-level language model, experimented with hyperparameters, and documented the codebase. Conducted training on the Shakespeare and Bible datasets, and provided comprehensive results analysis.
- **OpenAI's PyTorch and Transformer Resources**: Utilized OpenAI's guidance on PyTorch and transformer architectures for foundational understanding and implementation support.

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests to improve the project, add new datasets, or enhance functionality.

---

### Acknowledgments

- **Andrej Karpathy**: For his excellent course "Neural Networks: Zero to Hero," which inspired and laid the groundwork for the implementation of this character-level language model.
- **PyTorch Team**: For creating a versatile deep learning framework that powers this model's training and evaluation.
- **Shakespeare and Bible Text Datasets**: The Shakespeare and Bible texts provided historical and stylistic variety, essential for showcasing the model's adaptability across different language patterns.

