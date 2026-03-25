# Social Science Prompt Wizard - User Guide

## 🎓 Overview

The **Social Science Prompt Wizard** is an interactive tool designed to help researchers create sophisticated, structured annotation prompts for their social science projects. It guides you through a step-by-step process to define your research objectives, data characteristics, and annotation categories.

## 🚀 Key Features

- **Guided Workflow**: Step-by-step prompts for defining your annotation task
- **AI-Assisted Definitions**: Optional LLM assistance for generating category definitions
- **Dual Annotation Modes**: Support for both entity extraction and categorical annotation
- **Hierarchical Categories**: Create nested category structures (e.g., person → politician/scientist)
- **Example Generation**: Add example annotations to guide the LLM
- **Interactive Editing**: Review and manually edit the generated prompt
- **Auto-saving**: Option to save prompts for reuse

## 📋 How to Use

### Step 1: Access the Wizard

When running the LLMTool CLI, choose the wizard option when prompted for a prompt source:

```bash
python -m llm_tool
```

Then select:
- Mode: **Quick Start** or **Advanced**
- Prompt Mode: **simple** or **multi**
- Prompt Method: **wizard** ← Select this option!

### Step 2: AI Assistance (Optional)

The wizard will ask if you want AI assistance for generating category definitions:

```
🤖 Do you want AI assistance for generating category definitions? [Y/n]:
```

**Benefits of AI Assistance:**
- Automatically generates precise, clear definitions
- Saves time on definition writing
- Ensures consistency across definitions

**Available Models:**
- **Local**: Any Ollama model (llama3.3, gemma2, etc.)
- **Cloud**: OpenAI models (requires API key)

### Step 3: Project Description

Provide a brief description of your research project.

**Example:**
```
Analyzing political discourse in Canadian parliamentary debates to identify
policy themes, party positions, and sentiment towards various socio-economic
issues from 2015-2024.
```

### Step 4: Data Description

Describe the nature of your data to be annotated.

**Example:**
```
One-sentence excerpts from Hansard transcripts and Canadian news articles
(La Presse, Globe and Mail, CBC) discussing federal and provincial policies.
Texts are in English and French, typically 15-50 words, focusing on policy
statements and political commentary.
```

### Step 5: Choose Annotation Type

Select between two annotation strategies:

#### 1️⃣ Named Entity Extraction
Extract and classify specific entities or concepts from text.

**Use Cases:**
- Extracting person names and their roles
- Identifying organizations and their types
- Finding policy topics and their categories

**Example Output:**
```json
{
  "persons": ["Justin Trudeau", "Pierre Poilievre"],
  "person_role": "politician",
  "organizations": ["Liberal Party of Canada"],
  "org_type": "political_party"
}
```

#### 2️⃣ Categorical Annotation
Classify entire text into predefined categories.

**Use Cases:**
- Theme classification (environment, health, economy)
- Sentiment analysis (positive, neutral, negative)
- Party identification

**Example Output:**
```json
{
  "theme": "environment",
  "sentiment": "positive",
  "party": "LPC"
}
```

### Step 6: Define Categories

This is the core of the wizard. You'll define each category with:

#### For Categorical Annotation:

**Category Definition Process:**

1. **Category Name** (JSON key)
   - Example: `theme`, `sentiment`, `party`
   - Auto-sanitized (spaces → underscores, lowercase)

2. **General Description**
   - Example: "Primary policy theme discussed in the text"

3. **Define Values** (for each category)
   - Value name: `environment`
   - Definition (manual or AI-generated):
     - *Manual*: "Text relates to environmental policy, climate change, or ecological issues"
     - *AI-generated*: "Text discusses environmental protection, pollution control, climate change mitigation, conservation efforts, or ecological sustainability"

4. **Multiple Values?**
   - Can a text have multiple values for this category?
   - Example: `theme` might allow `["environment", "energy"]`

5. **Repeat** for each category

#### For Entity Extraction:

**Entity Definition Process:**

1. **Entity Category Name**
   - Example: `persons`, `organizations`

2. **Description**
   - Example: "Individuals mentioned in the text"

3. **Has Sub-types?**
   - Example: persons → politician, scientist, activist
   - If yes:
     - Parent category name: `person_role`
     - Define each sub-type with definitions

### Step 7: Add Examples (Optional)

Provide example texts and their correct annotations:

**Example 1:**
```
Text: "Pierre Poilievre criticized the new environmental regulations."

Annotations:
- theme: environment
- sentiment: negative
- party: CPC
```

These examples help the LLM understand your annotation requirements.

### Step 8: Review & Edit

The wizard generates a complete prompt. You can:

1. **✅ Accept** - Use as-is
2. **✏️ Edit** - Opens in your default text editor (EDITOR env variable)
3. **🔄 Regenerate** - Start over with modifications
4. **💾 Save** - Save to file for later use

### Step 9: Generated Prompt Structure

The final prompt includes:

```
1. Introduction & Role Definition
2. Task Description
3. Category Definitions (with all values)
4. Detailed Instructions
5. Examples (if provided)
6. Expected JSON Keys Template
```

## 💡 Best Practices

### 1. Clear, Specific Definitions

❌ **Poor Definition:**
```
"environment" - relates to environment
```

✅ **Good Definition:**
```
"environment" if the text relates to environmental policy, including
pollution control, climate change, conservation, species protection,
or ecological sustainability
```

### 2. Mutually Exclusive vs. Multi-Select

**Mutually Exclusive** (single value):
- Sentiment: positive OR neutral OR negative (not both)
- Use when categories don't overlap

**Multi-Select** (multiple values):
- Themes: ["environment", "energy", "economy"]
- Use when text can belong to multiple categories

### 3. Include "null" Options

Always include a "null" value for cases where the category doesn't apply:

```json
{
  "theme": "environment",
  "specific_policy": null,  ← No specific policy mentioned
  "sentiment": "positive"
}
```

### 4. Use Examples Strategically

Include examples that:
- Cover edge cases
- Show multiple category assignments
- Demonstrate "null" usage
- Represent your data's diversity

### 5. AI-Assisted Definitions

When using AI assistance:
- Review generated definitions carefully
- Edit if too verbose or unclear
- Ensure consistency across related categories
- Test with your actual data

## 🎯 Common Use Cases

### Use Case 1: Policy Theme Classification

**Goal**: Classify parliamentary debate excerpts by policy theme

**Configuration:**
- Annotation Type: Categorical
- Categories:
  - `theme`: environment, health, economy, education, etc.
  - `sentiment`: positive, neutral, negative
  - `party`: LPC, CPC, NDP, BQ, etc.

### Use Case 2: Named Entity Recognition

**Goal**: Extract persons and organizations with their roles

**Configuration:**
- Annotation Type: Entity Extraction
- Categories:
  - `persons`: (free-form names)
  - `person_role`: politician, scientist, activist, journalist
  - `organizations`: (free-form names)
  - `org_type`: government, ngo, corporation, media

### Use Case 3: Sentiment Analysis with Context

**Goal**: Analyze sentiment towards specific topics

**Configuration:**
- Annotation Type: Categorical
- Categories:
  - `topic`: (what the sentiment is about)
  - `sentiment`: positive, neutral, negative
  - `intensity`: low, medium, high

## 🔧 Troubleshooting

### Issue: AI Assistant Not Working

**Solution:**
1. Check if Ollama is running: `ollama list`
2. Verify model is available: `ollama pull llama3.3`
3. Try a different model
4. Continue without AI assistance (manual definitions)

### Issue: Generated Prompt Too Long

**Solution:**
- Reduce number of categories
- Shorten value definitions
- Remove some examples
- Use concise language

### Issue: LLM Not Following Prompt

**Solution:**
- Add more specific instructions
- Include clearer examples
- Use more precise value definitions
- Try a more capable model

### Issue: JSON Parsing Errors

**Solution:**
- Ensure all expected keys are defined
- Check for special characters in definitions
- Verify JSON template at end of prompt
- Test with simple examples first

## 📚 Advanced Tips

### Tip 1: Hierarchical Categories

Create parent-child category relationships:

```
Category: policy_type (parent)
Values: domestic, international, mixed

Category: specific_policy (child, depends on policy_type)
Values: healthcare, education, defense, trade, etc.
```

### Tip 2: Conditional Logic in Definitions

Use clear conditional language:

```
"welfare_state" if the text relates to social protection, elderly
assistance, or disability support AND mentions government programs
```

### Tip 3: Multi-Language Support

For multilingual data, include language indicators in definitions:

```
"health" if the text discusses "santé" (FR) or "health" (EN)
including healthcare system, medical coverage, or public health
```

### Tip 4: Save Reusable Prompts

After creating a good prompt:
1. Save it with a descriptive name
2. Store in `prompts/` directory
3. Reuse for similar projects
4. Create variations for different contexts

## 🎓 Example Session

Here's a complete example session:

```
=== Social Science Prompt Wizard ===

Step 1: Project Description
> Analyzing Canadian news articles for policy coverage

Step 2: Data Description
> News headlines from CBC, Globe and Mail (2020-2024)

Step 3: Annotation Type
> [2] Categorical Annotation

Step 4: Define Categories

Category #1
Name: policy_theme
Description: Main policy area discussed
Values:
  - environment: Environmental policy and climate issues
  - health: Healthcare system and public health
  - economy: Economic policy and financial issues
  - null: No clear policy theme

Multiple values? No

Category #2
Name: sentiment
Description: Tone toward the policy
Values:
  - positive: Favorable or supportive tone
  - neutral: Factual reporting without bias
  - negative: Critical or opposing tone

Multiple values? No

Add another category? No

Step 5: Examples
Add examples? Yes

Example #1
Text: "Government announces major investment in renewable energy"
policy_theme: environment
sentiment: positive

Add another example? No

Step 6: Review
[Displays generated prompt]

Accept? Yes

✓ Prompt generated successfully!
```

## 📖 Related Documentation

- [LLMTool Main Documentation](../README.md)
- [Annotation Pipeline Guide](annotation_pipeline.md)
- [Training Models Guide](model_training.md)
- [Prompt Best Practices](prompt_best_practices.md)

---

**Need Help?** Open an issue on GitHub or consult the examples in `prompts/` directory.
