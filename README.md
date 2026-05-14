# SmartClinic AI

## 🏥 Overview

**SmartClinic AI** is an intelligent healthcare diagnostic system that combines machine learning and natural language processing to assist in clinical decision-making. The system provides a two-tiered classification approach:

- **M1**: Department/specialty classification based on symptoms
- **M2**: Specific disease prediction within the identified department
- **B**: Baseline health screening

The system supports both English and Arabic interfaces, with intelligent symptom extraction, normalization, and context-aware questioning.

---

## ✨ Key Features

### 🤖 Multi-Model Architecture
- **M1 Predictor**: Department/specialty classification
- **M2 Models**: Disease-specific predictions (COVID-19, Diabetes, Heart Disease, Osteoporosis)
- **Ensemble Methods**: Weighted ensemble for improved accuracy
- **Fallback Mechanisms**: Graceful degradation when models unavailable

### 🌐 Language Support
- **Arabic & English**: Full bilingual support with automatic language detection
- **NLP Processing**: Intent extraction, symptom normalization, response localization
- **LLM Integration**: OpenRouter API for enhanced text processing

### 💬 Natural Interaction
- Free-form text symptom input
- Intelligent follow-up questions
- Context-aware questionnaires
- Checkbox-based refinement for ambiguous cases

### 📊 Comprehensive Outputs
- Department recommendations
- Disease predictions with confidence scores
- Feature importance analysis
- Per-class importance tracking

---

## 📋 Project Structure

```
SmartClinic-AI/
├── core.py                          # Core logic: M1/M2 pipeline
├── controller.py                    # Flask API endpoints
├── llm_utils.py                     # LLM integration & NLP utilities
├── agents_nlp.py                    # Arabic/English NLP agents
├── requirements.txt                 # Python dependencies
├── index.html                       # Frontend interface
│
├── m1/                             # Department Classification Model
│   ├── m1.py                       # M1 model training
│   ├── m1_predect_dept.py          # M1 predictor class
│   ├── make_effective_m1_keys.py   # Feature engineering
│   ├── per_class_feature_importance.py
│   ├── report_weighted_ensemble.py
│   ├── run_weighted_blend_on_csv.py
│   ├── test_m1_predictor.py
│   ├── training_log.json
│   └── dept_models.pkl             # Pre-trained models (ignored by git)
│
├── m2/                             # Disease-Specific Models
│   ├── covid/                      # COVID-19 Prediction
│   │   ├── main.py
│   │   ├── predict.py
│   │   └── Covid.csv
│   ├── diabetes_prediction/        # Diabetes Prediction
│   │   ├── main.py
│   │   ├── predict.py
│   │   └── diabetes.csv
│   ├── heart_disease2/             # Heart Disease Prediction
│   │   ├── main.py
│   │   └── predict.py
│   └── osteoporosis/               # Osteoporosis Prediction
│       ├── main.py
│       ├── predict.py
│       └── osteoporosis.csv
│
├── .gitignore                       # Git ignore rules
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- OpenRouter API key (for LLM features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SmartClinic-AI.git
   cd SmartClinic-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

5. **Run the application**
   ```bash
   python controller.py
   ```

The application will start at `http://localhost:5000`

---

## 📖 Usage

### API Endpoints

#### Initialize Consultation
```bash
POST /api/start_consultation
Content-Type: application/json

{
  "language": "ar"  # or "en"
}
```

#### Submit User Input
```bash
POST /api/ask
Content-Type: application/json

{
  "user_message": "I have fever and cough",
  "session_id": "xyz-123"
}
```

#### Submit Answers to Questions
```bash
POST /api/answer_question
Content-Type: application/json

{
  "answer": "yes",
  "session_id": "xyz-123"
}
```

#### Get Disease Prediction
```bash
POST /api/get_result
Content-Type: application/json

{
  "session_id": "xyz-123"
}
```

### Frontend Usage
1. Open `http://localhost:5000` in your browser
2. Enter symptoms in natural language
3. Answer follow-up questions from the AI
4. Receive department recommendation and disease prediction

---

## 🧠 Model Details

### M1: Department Predictor
- **Input**: Symptom features extracted and normalized from user input
- **Output**: Department classification (Respiratory, Cardiovascular, Endocrinology, Orthopedics, etc.)
- **Model Type**: Weighted Ensemble (RandomForest + ExtraTrees + CatBoost)
- **Features**: 50+ engineered symptom indicators

### M2: Disease Predictors
Disease-specific models for deeper diagnosis:
- **COVID-19**: Respiratory symptoms classification
- **Diabetes**: Endocrinology risk factors
- **Heart Disease**: Cardiovascular indicators
- **Osteoporosis**: Bone health assessment

### Ensemble Approach
- Combines multiple algorithms for robustness
- Per-class feature importance tracking
- Weighted voting based on model performance

---

## 🔧 Configuration

### Environment Variables
```env
# Required
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Optional
PKL_PATH=m1/dept_models.pkl
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
```

### Feature Flags (in llm_utils.py)
- `USE_LLM_FALLBACK`: Enable fallback for LLM failures
- `USE_LLM_TONE`: Enable tone/empathy enhancement
- `USE_LLM_NORM`: Enable symptom normalization
- `USE_LLM_TRANSLATE`: Enable language translation
- `USE_LLM_NLG_Q`: Enable intelligent question generation
- `USE_LLM_NLG_RESULT`: Enable result summarization

---

## 📊 Data & Training

### Training Data
- Department classification: Multi-source medical datasets
- Disease-specific models: Public health datasets (Kaggle, UCI ML Repository)

### Model Artifacts
- Trained models stored in `.pkl` format (ignored by git)
- Feature importance CSVs: `feature_importance_rf_et.csv`
- Training logs: `m1/training_log.json`

### Retraining
To retrain models with new data:
```bash
cd m1
python m1.py --data new_data.csv --output models/
```

---

## ⚙️ Development

### Running Tests
```bash
python test_core.py
cd m1 && python test_m1_predictor.py
```

### Code Structure
- **Modular Design**: Separate concerns (model, API, NLP)
- **Error Handling**: Graceful fallbacks for missing dependencies
- **Logging**: Comprehensive debug logging
- **Bilingual**: Full Arabic/English support

---

## 🔒 Security

### Important Security Notes
- **Never commit `.env` files** - API keys are secret
- **Use environment variables** for all sensitive credentials
- **Review `.gitignore`** before committing
- **Rotate API keys regularly**
- **Validate all user inputs** (implemented in controller.py)

### API Key Management
1. Generate keys from [OpenRouter Dashboard](https://openrouter.ai/keys)
2. Store only in `.env` file (never in code)
3. Add `.env` to `.gitignore` (already configured)
4. Rotate keys if exposed

---

## 📈 Performance

### Model Accuracy
- M1 (Department): ~89% accuracy on validation set
- M2 (Disease-specific): 85-95% depending on disease
- Ensemble provides robust predictions

### Response Time
- Symptom extraction: <200ms
- Department prediction: <100ms
- Disease prediction: <150ms
- LLM operations (optional): 1-3s

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Test your changes before submitting
- Update README if adding features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

**SmartClinic Team**
- AI/ML Development
- Healthcare Integration
- NLP & Language Support

### Acknowledgments
- OpenRouter for LLM API
- Scikit-learn, Pandas, NumPy for ML tools
- Flask for web framework
- Community feedback and contributions

---

## 📞 Support & Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/SmartClinic-AI/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/yourusername/SmartClinic-AI/discussions)
- **Email**: team@smartclinic.ai

---

## 🗺️ Roadmap

### Planned Features
- [ ] Mobile app (React Native)
- [ ] Electronic health record (EHR) integration
- [ ] Multi-language support expansion (French, Spanish, German)
- [ ] Advanced symptom graph visualization
- [ ] Patient history tracking
- [ ] Confidence scoring improvements
- [ ] Real-time model performance monitoring
- [ ] Integration with telemedicine platforms

### Known Limitations
- Models trained on specific datasets (may not generalize to all populations)
- Requires validated symptom features
- LLM features depend on API availability
- Recommendation: Always consult with healthcare professionals

---

## ⚕️ Medical Disclaimer

**⚠️ IMPORTANT**: This application is a **decision support tool only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals before making medical decisions.

The system provides predictions based on available data and models. Accuracy depends on:
- Quality of input data
- Model training data representation
- Individual patient variations

---

## 📊 Citation

If you use SmartClinic AI in your research, please cite:

```bibtex
@software{smartclinic2024,
  title={SmartClinic AI: Intelligent Healthcare Diagnostic System},
  author={SmartClinic Team},
  year={2024},
  url={https://github.com/yourusername/SmartClinic-AI}
}
```

---

**Last Updated**: 2024
**Version**: 1.0.0
