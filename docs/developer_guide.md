# Developer Guide

This guide provides detailed information for developers working on the CurioScan project.

## Development Environment Setup

### Recommended Tools

- **IDE**: Visual Studio Code with Python, Docker, and YAML extensions
- **Python Version**: 3.10+
- **Virtual Environment**: `venv` or `conda`
- **Docker**: Latest version with Docker Compose
- **Git**: Latest version

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sagexd08/Ocr-Model.git
   cd Ocr-Model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

```
├── alembic/              # Database migrations
├── api/                  # FastAPI application
│   ├── routers/          # API endpoint routers
│   ├── dependencies.py   # Dependency injection
│   ├── main.py           # API entry point
│   └── models.py         # Data models
├── configs/              # Configuration files
├── data/                 # Data storage directory
├── docs/                 # Documentation
├── models/               # Model definitions
├── streamlit_demo/       # Streamlit demo application
├── tests/                # Test suite
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
├── worker/               # Celery worker
│   ├── pipeline/         # Processing pipeline
│   │   ├── processors/   # Pipeline processor modules
│   │   └── utils/        # Pipeline utilities
│   ├── celery_app.py     # Celery application
│   └── tasks.py          # Task definitions
├── .env.example          # Example environment variables
├── docker-compose.yml    # Docker Compose configuration
├── Makefile              # Make commands
├── README.md             # Project README
└── requirements.txt      # Project dependencies
```

## Development Workflow

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make changes and run tests:
   ```bash
   # Run unit tests
   pytest tests/unit

   # Run integration tests
   pytest tests/integration
   ```

3. Format and lint your code:
   ```bash
   # Format code
   make format

   # Run linting
   make lint
   ```

4. Commit changes:
   ```bash
   git commit -m "Add my new feature"
   ```

5. Push branch and create a pull request:
   ```bash
   git push origin feature/my-new-feature
   ```

### Adding a New Pipeline Processor

1. Create a new processor file in `worker/pipeline/processors/`:
   ```python
   # worker/pipeline/processors/my_processor.py
   from typing import Dict, Any
   from ...types import Document
   from ...utils.logging import get_logger, log_execution_time

   logger = get_logger(__name__)

   class MyProcessor:
       """
       My new document processor.
       """
       
       def __init__(self, config: Dict[str, Any] = None):
           self.config = config or {}
           # Initialize processor-specific settings
           
       @log_execution_time
       def process(self, document: Document) -> Document:
           """
           Process a document.
           
           Args:
               document: Document to process
               
           Returns:
               Processed document
           """
           logger.info(f"Processing document with {len(document.pages)} pages")
           
           # Implementation here
           
           return document
   ```

2. Register your processor in `worker/pipeline/pipeline_builder.py`

3. Add tests for your processor in `tests/unit/test_processors.py`

4. Add documentation for your processor in `docs/pipeline_processors.md`

### Adding a New API Endpoint

1. Create or modify a router file in `api/routers/`:
   ```python
   # api/routers/my_feature.py
   from fastapi import APIRouter, Depends, HTTPException, status
   from ..dependencies import get_db, get_current_user
   from ..models import MyFeatureRequest, MyFeatureResponse

   router = APIRouter(
       prefix="/my-feature",
       tags=["my-feature"],
       dependencies=[Depends(get_current_user)],
       responses={404: {"description": "Not found"}},
   )

   @router.post("/", response_model=MyFeatureResponse)
   async def create_my_feature(
       request: MyFeatureRequest,
       db=Depends(get_db),
       current_user=Depends(get_current_user)
   ):
       # Implementation here
       return {"status": "success", "data": {}}
   ```

2. Register the router in `api/main.py`:
   ```python
   from .routers import my_feature

   app.include_router(my_feature.router)
   ```

3. Add tests for your endpoint in `tests/unit/test_api.py`

4. Update the API documentation in `docs/api_reference.md`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run a specific test file
pytest tests/unit/test_processors.py

# Run a specific test
pytest tests/unit/test_processors.py::test_my_processor
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=worker --cov=api --cov-report=html

# View the report
open htmlcov/index.html
```

## Performance Profiling

### Profiling the Pipeline

```python
from worker.pipeline.pipeline_builder import PipelineBuilder
from worker.utils.profiling import profile_pipeline

# Create a pipeline
pipeline = PipelineBuilder.build(profile="default")

# Profile the pipeline
profiling_result = profile_pipeline(pipeline, document, iterations=10)
print(profiling_result.summary())
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory_profiler

# Run with memory profiling
python -m memory_profiler worker/tasks.py
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

### Updating API Reference

After adding or modifying API endpoints, update the API reference:

```bash
# Generate OpenAPI spec
python -c "from api.main import app; print(app.openapi())" > docs/openapi.json

# Update API reference (requires npm and redoc-cli)
npx redoc-cli bundle docs/openapi.json -o docs/api-reference.html
```

## Troubleshooting

### Common Issues

1. **Celery worker not starting**
   - Check Redis connection
   - Verify Celery configuration
   - Ensure broker URL is correct

2. **Document processing fails**
   - Check input document format
   - Verify model paths
   - Check storage permissions

3. **API returns 500 error**
   - Check API logs
   - Verify database connection
   - Check request payload

### Debugging

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug specific module
logger = logging.getLogger("worker.pipeline.processors")
logger.setLevel(logging.DEBUG)
```

## Performance Optimization

### Improving Processing Speed

1. Use GPU acceleration when available
2. Optimize image preprocessing
3. Use batched processing
4. Tune worker concurrency

### Memory Usage

1. Process large documents in chunks
2. Dispose of intermediate results
3. Use streaming responses for large files

## Contributing Guidelines

1. Follow the code style (PEP 8)
2. Write tests for new features
3. Update documentation
4. Keep pull requests focused on a single feature/fix
5. Add clear commit messages
