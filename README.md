# ETHOS - The AI Arbiter of Rational Discourse

## 🔍 Model Description

**ETHOS (Evaluative Text Heuristic for Objective Scrutiny)** is a pioneering argument analysis system that evaluates the logical structure, validity, and rhetorical quality of text. Designed to serve as an AI arbiter of rational discourse, ETHOS provides comprehensive analysis of argumentation quality through sophisticated NLP techniques.

Using a combination of rule-based analysis and neural language models (BERT), ETHOS dissects texts to identify claims, premises, logical fallacies, and rhetorical devices—creating a complete picture of argument structure and quality. The system then visualizes these relationships and provides detailed assessments of logical soundness and rhetorical clarity.

### 🌟 Key Features

- **🧩 Advanced Argument Structure Analysis**: Extracts claims and premises from text with high accuracy using dependency parsing and linguistic markers
  
- **🔗 Semantic Argument Linking**: Uses BERT embeddings to establish relationships between premises and claims based on semantic similarity
  
- **⚠️ Sophisticated Fallacy Detection**: Identifies common logical fallacies (Ad Populum, Hasty Generalization, False Dichotomy) through both rule-based and ML-based approaches
  
- **🎭 Rhetoric Device Recognition**: Detects rhetorical questions, strong sentiment expressions, and superlatives that may influence argument perception
  
- **📊 Logical Soundness Scoring**: Evaluates the overall logical quality of arguments with nuanced ratings
  
- **📈 Comprehensive Visualization**: Generates clear argument graphs showing support relationships between premises and claims
  
- **💬 Natural Language Explanations**: Provides human-readable analysis of logical structure and potential issues

## 🚀 Why ETHOS is Revolutionary

ETHOS represents a significant advancement in automated argument analysis:

1. **Deeper than Syntax**: Goes beyond simple keyword matching by using semantic understanding to link arguments
   
2. **Multi-dimensional Analysis**: Examines both structural (argumentation form) and rhetorical (persuasive techniques) aspects simultaneously
   
3. **Visual Representation**: Transforms complex argumentative structures into intuitive graphs
   
4. **Balanced Assessment**: Provides nuanced evaluation rather than binary judgments about argument quality
   
5. **Transparency**: Clearly highlights the evidence for each assessment, allowing users to understand the system's reasoning

## 💡 Intended Uses

ETHOS is designed for applications requiring sophisticated argument analysis:

- **📝 Academic Research**: Analyze argument structures in papers, essays, and scholarly works
  
- **🎓 Educational Assessment**: Evaluate student essays and argumentation skills
  
- **📰 Media Analysis**: Assess the logical quality of news articles, editorials, and opinion pieces
  
- **💼 Professional Writing**: Improve the logical coherence and persuasiveness of business communications
  
- **🗣️ Debate Preparation**: Analyze strengths and weaknesses in argumentative positions
  
- **📊 Policy Evaluation**: Examine the logical structure of policy proposals and position papers

## 🛠️ System Architecture

ETHOS consists of several modular components that work together seamlessly:

1. **argument_analyzer**: Extracts argument components (claims and premises) using dependency parsing and linguistic indicators
   
2. **logic_analyzer**: Identifies logical fallacies through rule-based patterns and machine learning
   
3. **rhetoric_analyzer**: Detects rhetorical devices using sentiment analysis and linguistic patterns
   
4. **argument_visualizer**: Creates graph-based visualizations of argument structure using semantic similarity
   
5. **synthesis_engine**: Combines all analyses to produce overall evaluations of argument quality
   
6. **nlp_utils**: Provides NLP infrastructure including spaCy and BERT embedding functionality

## 📊 Performance Highlights

ETHOS demonstrates strong capabilities in:

- **Argument Component Identification**: Accurately extracts claims and premises with confidence scores
- **Semantic Relationship Detection**: Successfully links related premises and claims based on meaning
- **Fallacy Recognition**: Identifies common fallacies with explanatory context
- **Rhetorical Pattern Detection**: Effectively flags potential rhetorical techniques
- **Intuitive Visualization**: Generates clear argument graphs showing support relationships

## 💻 Example Usage

```python
import spacy
import nlp_utils
import argument_analyzer
import logic_analyzer
import rhetoric_analyzer
import argument_visualizer
import synthesis_engine

# Load models
nlp_utils.load_spacy_model()
nlp_utils.load_bert()

# Process text
text = "Climate change is definitely happening because global temperatures have risen significantly over the past century. Everyone knows this is a serious problem that requires immediate action."
doc = nlp_utils.process_text_spacy(text)

# Get sentence embeddings for semantic linking
sentence_embeddings = nlp_utils.get_all_sentence_embeddings(doc)

# Analyze arguments
argument_components = argument_analyzer.enhanced_component_analyzer(doc)

# Detect fallacies
fallacy_findings = logic_analyzer.enhanced_fallacy_analyzer(doc)

# Analyze rhetoric
rhetoric_findings = rhetoric_analyzer.simple_rhetoric_analyzer(doc)

# Combine findings
all_findings = fallacy_findings + rhetoric_findings

# Generate summary ratings
analysis_summary = synthesis_engine.generate_summary_ratings(argument_components, all_findings)

# Create argument graph
argument_graph = argument_visualizer.build_argument_graph(argument_components, sentence_embeddings)

# Print results
print("Argument Components:")
for comp in argument_components:
    print(f"{comp.component_type}: {comp.text} (Confidence: {comp.confidence:.2f})")

print("\nFindings:")
for finding in all_findings:
    print(f"{finding.finding_type}: {finding.description}")

print("\nSummary Ratings:")
for category, rating in analysis_summary.items():
    print(f"{category}: {rating}")
```

## 🔧 Customization Options

ETHOS can be customized in several ways:

- **Fallacy Detection Rules**: Extend or modify rules for identifying additional fallacy types
- **Linguistic Indicators**: Add new claim/premise indicators for different domains or languages
- **Similarity Threshold**: Adjust the semantic similarity threshold for argument linking
- **Visualization Format**: Customize the representation of argument structures

## 📈 Future Development

The ETHOS project is continuously evolving, with planned enhancements including:

- **Expanded Fallacy Detection**: Additional fallacy types and improved ML-based detection
- **Cross-Document Analysis**: Analyze arguments spanning multiple documents
- **Multi-language Support**: Extend capabilities to additional languages
- **Interactive Visualizations**: Enhanced visual representations with interactive elements
- **API Implementation**: RESTful API for seamless integration with other applications
