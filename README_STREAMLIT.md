# 🔍 OCR System - Streamlit Web Interface

A complete web-based interface for the OCR Document Processing System built with Streamlit.

## 🚀 Quick Start

### Method 1: Using the Launcher Script
```bash
python run_streamlit.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run streamlit_app.py
```

### Method 3: Manual Setup
```bash
# Install dependencies
pip install streamlit plotly pandas

# Launch application
streamlit run streamlit_app.py --server.port 8501
```

## 🌐 Access the Application

Once started, open your web browser and navigate to:
- **Local URL:** http://localhost:8501
- **Network URL:** http://[your-ip]:8501 (for network access)

## 📋 Features

### 🔧 **Processing Settings**
- **Profile Selection:** Performance, Quality, or Default processing modes
- **Processing Mode:** Advanced, Standard, or Fast extraction
- **Page Limit:** Control how many pages to process (1-50)

### 📤 **Document Upload & Processing**
- **Supported Formats:** PDF, PNG, JPG, JPEG, TIFF, BMP
- **File Size Limit:** Up to 200MB per file
- **Real-time Processing:** Live progress updates and status
- **Batch Processing:** Process multiple documents sequentially

### 📊 **Results Visualization**
- **Text Extraction:** View extracted text with formatting
- **Confidence Analysis:** OCR confidence score distributions
- **Token Details:** Detailed token information with bounding boxes
- **Performance Metrics:** Processing time and accuracy statistics

### 📈 **Processing History**
- **Session History:** Track all processed documents
- **Performance Trends:** Processing time analysis over time
- **Statistics Dashboard:** Aggregate metrics and insights
- **Export Capabilities:** Download results in JSON format

## 🎯 Usage Guide

### Step 1: Initialize System
1. Click "🚀 Initialize OCR System" in the sidebar
2. Wait for the system to load (may take 30-60 seconds)
3. Look for "✅ OCR System Ready!" confirmation

### Step 2: Configure Settings
1. Select processing profile (Performance recommended for speed)
2. Choose processing mode (Advanced for best results)
3. Set max pages to process (1 for quick testing)

### Step 3: Upload Document
1. Go to "📤 Upload & Process" tab
2. Drag and drop or browse for your document
3. Review file information and settings
4. Click "🚀 Process Document"

### Step 4: View Results
1. Switch to "📊 Results" tab after processing
2. Review extracted text and confidence scores
3. Analyze token details and statistics
4. Export results if needed

### Step 5: Track History
1. Use "📈 History" tab to view all processed documents
2. Monitor processing performance trends
3. Compare results across different settings

## 🔧 Configuration

### Streamlit Settings
Edit `.streamlit/config.toml` to customize:
- Server port and host settings
- Upload file size limits
- Theme and appearance
- Logging levels

### OCR System Settings
The web interface uses the same configuration as the CLI:
- Pipeline configurations in `configs/pipeline_config.py`
- Model settings in `worker/model_manager.py`
- Processing parameters via the web UI

## 📊 Performance Tips

### For Best Speed:
- Use "Performance" profile
- Set max_pages to 1-5 for testing
- Use "Fast" mode for quick extraction
- Process smaller files first

### For Best Accuracy:
- Use "Quality" profile
- Use "Advanced" mode
- Allow more processing time
- Ensure good image quality

### For Production Use:
- Use "Default" profile for balanced results
- Monitor processing history for optimization
- Set appropriate page limits based on needs
- Regular system monitoring via the dashboard

## 🐛 Troubleshooting

### Common Issues:

**"OCR System initialization failed"**
- Ensure all dependencies are installed
- Check that PaddleOCR models can download
- Verify sufficient disk space and memory

**"File upload failed"**
- Check file size (max 200MB)
- Ensure supported file format
- Try refreshing the page

**"Processing takes too long"**
- Reduce max_pages setting
- Use "Performance" profile
- Try "Fast" mode for quick results

**"No text extracted"**
- Check if document contains actual text/images
- Try different processing modes
- Verify document is not corrupted

### System Requirements:
- **RAM:** 8GB+ recommended (4GB minimum)
- **Storage:** 5GB+ free space for models
- **CPU:** Multi-core processor recommended
- **Network:** Internet connection for initial model download

## 📁 File Structure

```
OCR(Freelance)/
├── streamlit_app.py          # Main Streamlit application
├── run_streamlit.py          # Launcher script
├── requirements_streamlit.txt # Streamlit dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── worker/                   # OCR system core
├── configs/                  # Pipeline configurations
├── models/                   # OCR models and adapters
└── output/                   # Processing results
```

## 🔒 Security Notes

- The application runs locally by default (localhost:8501)
- No data is sent to external servers
- All processing happens on your machine
- Uploaded files are temporarily stored and cleaned up
- No persistent storage of sensitive documents

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Try restarting the application

## 🎉 Enjoy Your OCR System!

The Streamlit interface provides a user-friendly way to interact with the powerful OCR system. Upload documents, experiment with settings, and analyze results all through an intuitive web interface.
