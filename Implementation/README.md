# IntelliPdM Predictive Maintenance System

A fog computing-based predictive maintenance system using machine learning for industrial equipment monitoring. The system provides both Java and Python implementations for flexible deployment options.

## System Architecture

- **Edge Processing**: ANN (Artificial Neural Network) for fast, low-latency predictions
- **Cloud Processing**: Random Forest for complex cases requiring higher accuracy
- **Hybrid Approach**: Intelligent offloading based on prediction confidence
- **Multi-Platform**: Java standalone and Python integration options

## Quick Start Guide

### Java Simulation (Standalone)
```bash
# Navigate to source directory
cd Implementation/src

# Compile Java files
javac -cp . intellipdm/*.java

# Run Java simulation
java intellipdm.JavaIntelliPdM
```

### Python Simulation (Full ML Pipeline)
```bash
# Navigate to Implementation directory
cd Implementation

# Run fog computing simulation
python fog_computing_simulation.py

# Or run basic simulation
python intellipdm_simulation.py
```

## Implementation Options

### Java Implementation (JavaIntelliPdM.java)
- **Advantages**: No external dependencies, faster startup, single JVM
- **Use Case**: Production deployment, embedded systems, standalone operation
- **Models**: Native Java ANN and Random Forest implementations
- **Dependencies**: Java 17+

## Running Simulations

### Java Simulation

#### Java Implementation
```bash
cd Implementation/src
javac -cp . intellipdm/*.java
java intellipdm.JavaIntelliPdM
```

**Features:**
- ✅ No external dependencies
- ✅ Built-in ANN and Random Forest models
- ✅ Realistic sensor data generation
- ✅ Multi-threaded machine monitoring
- ✅ Real-time fault detection simulation

### Python Simulations

#### Option 1: Continuous Fog Computing Simulation
```bash
cd Implementation
python fog_computing_simulation.py
```

**Features:**
- ✅ Continuous monitoring with real-time timestamps
- ✅ Multi-threaded machine monitoring
- ✅ 30 seconds of continuous operation
- ✅ 3-second monitoring intervals
- ✅ Real-time logging and results

#### Option 2: Batch Processing Simulation
```bash
cd Implementation
python intellipdm_simulation.py
```

**Features:**
- ✅ Batch processing of test cases
- ✅ Sequential machine processing
- ✅ Fixed number of predictions per machine
- ✅ Summary results after completion

#### Option 3: Direct Python ML Scripts
```bash
cd Implementation

# Train models
python python_ml/train_ann.py
python python_ml/train_rf.py
python python_ml/train_cnn.py

# Make predictions
echo '{"temp": 65.0, "voltage": 245.0, "vibration": [0.1, 0.2, 0.3]}' > test.json
python python_ml/predict_ann.py test.json
python python_ml/predict_rf.py test.json
```

## Machine Learning Models

### 1. ANN (Edge Processing)
- **Implementation**: TensorFlow/Keras (Python) or Native Java
- **Purpose**: Fast edge processing with ~350ms latency
- **Features**: 7 features (temp, voltage, vibration statistics)
- **Files**: 
  - Python: `python_ml/ann_model.keras`, `python_ml/predict_ann.py`
  - Java: Built into `JavaIntelliPdM.java`

### 2. Random Forest (Cloud Processing)
- **Implementation**: scikit-learn (Python) or Native Java
- **Purpose**: Higher accuracy cloud processing
- **Features**: 6 features (temp, voltage, vibration components, current)
- **Files**: 
  - Python: `python_ml/rf_model.pkl`, `python_ml/predict_rf.py`
  - Java: Built into `JavaIntelliPdM.java`

### 3. CNN (Reference Implementation)
- **Implementation**: TensorFlow/Keras (Python only)
- **Purpose**: Research and comparison baseline
- **Features**: Raw vibration time series (100 points)
- **Files**: `python_ml/cnn_model.h5`, `python_ml/predict_cnn.py`

## Configuration

### config.properties
```properties
# Machine Configuration
num_machines=5
simulation_time_seconds=30
monitor_interval_seconds=3

# ML Configuration
edge_confidence_threshold=0.7
python_executable=python

# Model Paths (Python integration)
ann_model_path=python_ml/ann_model.keras
rf_model_path=python_ml/rf_model.pkl
scaler_path=python_ml/ann_scaler.joblib
```

## Performance Metrics

### Expected Results
- **Overall Accuracy**: ~72-85%
- **Edge Processing**: 85-96% of predictions handled at edge
- **Cloud Processing**: 4-15% offloaded for complex cases
- **Edge Latency**: ~1-5ms (Java) / ~350ms (Python)
- **Cloud Latency**: ~2-8ms (Java) / ~985ms (Python)

### Java vs Python Performance
| Metric | Java Implementation | Python Integration |
|--------|--------------------|--------------------|
| Startup Time | < 1 second | 3-5 seconds |
| Edge Latency | 1-5ms | 300-400ms |
| Cloud Latency | 2-8ms | 800-1000ms |
| Memory Usage | Lower | Higher |
| Dependencies | Java 17+ only | Python + ML libs |
| Model Accuracy | Simplified | Full ML pipeline |

## File Structure

```
Implementation/
├── README.md                        # This file
├── config.properties               # Configuration file
│
├── src/intellipdm/                 # Java implementation
│   └── JavaIntelliPdM.java        # Standalone Java simulation
│
├── python_ml/                     # Python ML pipeline
│   ├── Models:
│   │   ├── ann_model.keras         # ANN model for edge
│   │   ├── rf_model.pkl           # Random Forest for cloud
│   │   ├── cnn_model.h5           # CNN reference model
│   │   ├── ann_scaler.joblib      # Feature scaling
│   │   └── ann_imputer.joblib     # Data imputation
│   │
│   ├── Training Scripts:
│   │   ├── train_ann.py           # Train ANN model
│   │   ├── train_rf.py            # Train Random Forest
│   │   └── train_cnn.py           # Train CNN model
│   │
│   ├── Prediction Scripts:
│   │   ├── predict_ann.py         # ANN predictions
│   │   ├── predict_ann_fast.py    # Optimized ANN
│   │   ├── predict_ultra_fast.py  # Rule-based predictions
│   │   ├── predict_rf.py          # Random Forest predictions
│   │   └── predict_cnn.py         # CNN predictions
│   │
│   ├── Data:
│   │   ├── clean_synthetic_data.csv # Training dataset
│   │   ├── test_fault_input.json   # Test fault case
│   │   └── test_normal_input.json  # Test normal case
│   │
│   └── Utilities:
│       ├── generate_clean_data.py  # Data preprocessing
│       ├── analyze_data.py         # Data analysis
│       └── test_model.py          # Model testing
│
├── Simulation Scripts:
│   ├── fog_computing_simulation.py    # Main fog simulation
│   ├── intellipdm_simulation.py       # Basic simulation
│   ├── simulate_intellipdm.py         # Alternative simulation
│   └── prediction_server.py           # Prediction service
│
├── Testing & Analysis:
│   ├── test_speed.py                  # Speed comparisons
│   ├── comprehensive_speed_test.py    # Full speed analysis
│   ├── test_model_accuracy.py         # Accuracy testing
│   ├── profile_latency.py             # Latency profiling
│   └── full_latency_analysis.py       # Complete latency analysis
│
└── Configuration & Data:
    ├── dataset/                       # Fog computing datasets
    ├── results/                       # Simulation results
    ├── output/                        # Generated outputs
    └── jars/                         # Java dependencies
```

## System Requirements

### For Java Implementation
- **Java**: 17+ (OpenJDK or Oracle JDK)
- **Memory**: 512MB RAM minimum
- **OS**: Windows, Linux, macOS

### For Python Integration (Optional)
- **Java**: 17+ 
- **Python**: 3.8+ 
- **Libraries**: TensorFlow 2.x, scikit-learn, pandas, numpy
- **Memory**: 2GB RAM recommended
- **OS**: Windows, Linux, macOS

## Installation

### Java Setup
```bash
# Verify Java installation
java -version

# Compile if needed
cd Implementation/src
javac -cp . intellipdm/*.java
```

### Python Setup
```bash
# Install Python dependencies
pip install tensorflow scikit-learn pandas numpy joblib

# Verify installation
python -c "import tensorflow, sklearn, pandas, numpy; print('All libraries installed')"
```

## Troubleshooting

### Common Issues

1. **Java ClassNotFound**: Ensure you're in the `src` directory when running
2. **Python not found**: Check Python is in system PATH
3. **Model files missing**: Run training scripts first
4. **Permission denied**: Ensure write permissions for output directories

### Performance Tuning

1. **Java heap size**: Use `-Xmx2g` for large simulations
2. **Thread count**: Adjust based on available CPU cores
3. **Monitoring interval**: Increase for better performance, decrease for more data

## Use Cases

### Production Deployment
- Use **Java Implementation** for reliability and performance
- Deploy on edge devices or industrial controllers
- No external dependencies required

### Research & Development
- Use **Python Scripts** for full ML pipeline
- Experiment with different models and algorithms
- Access to complete data science ecosystem

### Hybrid Deployment
- Use **Java** for production edge processing
- Use **Python** for model training and experimentation
- Best of both worlds approach

This system demonstrates the effectiveness of fog computing for industrial IoT applications, providing both low-latency edge processing and high-accuracy cloud processing as needed.
