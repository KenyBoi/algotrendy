# AI Orchestrator Retool Dashboard

This directory contains the Retool dashboard configuration for monitoring and managing the AI Orchestrator Module in AlgoTrendy.

## 🚀 Quick Start

### 1. Start the AI Orchestrator API

First, make sure you have all dependencies installed:

```bash
cd src
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python ai_orchestrator_api.py
```

The API will be available at `http://localhost:8000`

### 2. Import Dashboard into Retool

1. Open your Retool account
2. Create a new app
3. Go to the app settings and import the dashboard configuration:
   - Click on "Import" in the top right
   - Upload the `ai_orchestrator_dashboard.json` file
4. Configure the API connection:
   - Go to Resources → Add Resource
   - Select "REST API"
   - Set Base URL to `http://localhost:8000`
   - Name it "AI Orchestrator API"

### 3. Set Environment Variables

Make sure your environment variables are set for the AI providers:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GITHUB_TOKEN="your-github-token"
```

## 📊 Dashboard Features

### Dashboard Overview Page

- **Real-time Metrics**: Total queries, costs, active providers, and average response times
- **Provider Status Table**: Live status of all AI providers with health indicators
- **Usage Charts**: Visual representation of provider usage distribution
- **Health Check Actions**: Manually trigger health checks for individual providers

### Query Interface Page

- **AI Query Form**: Submit queries to the orchestrator with intelligent provider selection
- **Query Types**: Choose from analysis, strategy, conversation, code generation, etc.
- **Cost Control**: Set maximum cost limits per query
- **Real-time Results**: View responses with confidence scores, costs, and processing times

### Provider Comparison Page

- **Multi-Provider Comparison**: Compare responses from all AI providers simultaneously
- **Consensus Scoring**: See which provider gives the most consistent results
- **Detailed Metrics**: Compare confidence, cost, and processing time across providers
- **Best Provider Selection**: Automatic identification of the highest-confidence response

## 🔧 API Endpoints

The dashboard connects to these REST API endpoints:

- `GET /health` - Health check
- `GET /metrics` - Orchestrator metrics
- `GET /providers` - Provider status
- `POST /query` - Submit AI query
- `POST /compare` - Compare providers
- `POST /providers/{name}/health-check` - Manual health check

## 🎨 Dashboard Components

### Stats Cards
Display key metrics with icons and color coding:
- 🟢 Healthy providers
- 🟡 Degraded providers
- 🔴 Unhealthy providers

### Interactive Tables
- Sortable columns
- Real-time data updates
- Action buttons for health checks

### Charts
- Bar charts for usage distribution
- Real-time updates every 30 seconds

### Forms
- Input validation
- Dynamic query type selection
- Cost limit controls

## 🔒 Security Considerations

For production deployment:

1. **API Authentication**: Add API key authentication to the FastAPI endpoints
2. **HTTPS**: Use HTTPS for all API communications
3. **Rate Limiting**: Implement rate limiting on the API endpoints
4. **Environment Variables**: Never commit API keys to version control
5. **CORS**: Restrict CORS origins to your Retool domain only

## 🚀 Production Deployment

### API Server
```bash
# Using uvicorn with production settings
uvicorn ai_orchestrator_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "ai_orchestrator_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GITHUB_TOKEN=your-token-here
REDIS_URL=redis://your-redis-instance:6379
API_SECRET_KEY=your-secret-key
```

## 📈 Monitoring & Analytics

The dashboard provides insights into:

- **Provider Performance**: Response times, success rates, costs
- **Query Patterns**: Most used query types and providers
- **Cost Analysis**: Total spending and per-query costs
- **System Health**: Real-time provider availability

## 🛠️ Customization

### Adding New Providers
1. Add the provider to the AI Orchestrator
2. Update the API endpoints
3. Modify the dashboard JSON to include new provider metrics

### Custom Query Types
1. Add new query types to the `QueryType` enum
2. Update provider routing logic
3. Dashboard will automatically show new query types

### Additional Metrics
1. Extend the metrics collection in the orchestrator
2. Update the API response models
3. Add new dashboard components

## 📚 API Documentation

Full API documentation is available at `http://localhost:8000/docs` when the server is running (Swagger UI).

## 🤝 Contributing

When updating the dashboard:

1. Export the updated configuration from Retool
2. Update the JSON file in this directory
3. Test the import in a fresh Retool app
4. Document any new features or changes

## 🆘 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check if the API server is running on port 8000
   - Verify CORS settings allow your Retool domain

2. **Provider Status Shows Offline**
   - Check environment variables are set correctly
   - Verify API keys are valid and have sufficient credits

3. **Queries Timeout**
   - Check network connectivity to AI provider APIs
   - Verify rate limits haven't been exceeded

4. **Dashboard Not Loading**
   - Clear browser cache
   - Check Retool app permissions
   - Verify JSON configuration is valid

### Logs

API server logs are available in the terminal where the server is running. Enable debug logging by setting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)