# Deploying VectorFin API on Vercel

This guide explains how to deploy the VectorFin API on Vercel's serverless platform.

## Prerequisites

1. A [Vercel account](https://vercel.com/signup)
2. [Vercel CLI](https://vercel.com/docs/cli) installed (optional for local testing)
3. Git repository with your VectorFin code

## Deployment Steps

### 1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)

Make sure your code is pushed to a Git repository that Vercel can access.

### 2. Connect to Vercel

Go to [Vercel's dashboard](https://vercel.com/dashboard) and click "Add New..." → "Project".

### 3. Import your repository

Select the repository containing your VectorFin code.

### 4. Configure the project

- **Framework Preset**: Select "Other"
- **Root Directory**: Leave as is (default)
- **Build Command**: Leave as is (default)
- **Output Directory**: Leave as is (default)

### 5. Environment Variables

Add the following environment variables:

#### Required Environment Variables

```bash
# Required - API key for fetching financial news data
# You can use one of the alternatives listed in the "Free Alternatives" section below
NEWS_API_KEY=your_newsapi_key_here
```

#### Free Alternatives to NewsAPI

Here are some free alternatives to NewsAPI that you can use for financial news data:

1. **Alpha Vantage News API**:
   - Free tier includes basic news endpoints
   - Offers 25 API requests per day for free
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Use the environment variable: `ALPHA_VANTAGE_API_KEY=your_key_here`
   - API endpoint: `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=your_key_here`

2. **Finnhub**:
   - Free tier includes news API with some limitations
   - 60 API calls per minute
   - Sign up at [Finnhub](https://finnhub.io/register)
   - Use the environment variable: `FINNHUB_API_KEY=your_key_here`

3. **Yahoo Finance API (via RapidAPI)**:
   - Several plans including a limited free tier
   - Access via [RapidAPI](https://rapidapi.com/apidojo/api/yahoo-finance1)

You will need to modify the VectorFin code to use these alternative APIs, but the basic structure should be similar.

#### Optional Environment Variables

Default values will be used if these are not set:

```bash
# Optional - Default is the local endpoint value from your config
LLM_API_URL=your_llm_api_url_here

# Optional - Default is "gemma-3-4b-it-qat" or value from your config
INTERPRETATION_MODEL=your_model_name_here

# Optional - For LLM services requiring authentication
LLM_API_KEY=your_llm_api_key_here

# Optional - A random secure key will be generated if not provided
# Note: This key will be shown in logs on initial deployment
ADMIN_API_KEY=your_custom_admin_key_here
```

### 6. Deploy

Click "Deploy" and wait for the deployment to complete.

## Important Notes for Vercel Deployment

1. **File Storage**: Vercel's functions are stateless and don't provide persistent file storage. Any files written by your API (like prediction outputs) won't persist between function invocations. Consider using a database or storage service instead.

2. **Cold Starts**: Be aware of cold starts with serverless functions. The first request after a period of inactivity may take longer to process.

3. **Execution Time Limits**: Vercel has execution time limits for functions (30 seconds by default). If your predictions take longer, consider optimizing or using background processing.

4. **Memory Limitations**: Vercel functions have memory limits. If your models require significant memory, you might need to optimize or consider other deployment options.

5. **API Key Security**: Make sure to use environment variables for all sensitive keys and credentials.

## How Default Values Work

When deploying to Vercel, the system will use default values for certain configurations if they're not explicitly set:

1. **LLM Configuration**: If no LLM_API_URL or INTERPRETATION_MODEL is provided, the system will use the default values from `vectorfin/src/utils/config.py` (typically a local LLM endpoint).

2. **API Keys**: If no ADMIN_API_KEY is provided, a secure random key will be generated during deployment and shown in the deployment logs. Make sure to save this key.

3. **Rate Limits**: Default rate limits will be applied to API users (100 requests per day for regular users, 1000 for admin).

4. **Models Directory**: The system will use the default models directory path from your config.

## Testing Your Deployment

After deployment, you can test your API using the provided URL:

```bash
curl -X POST "https://your-vercel-app.vercel.app/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "prediction_horizon": 5
  }'
```

## Using Vercel's Fluid Compute

For better performance, enable Fluid Compute in your Vercel project settings:

1. Go to Vercel Dashboard → Project Settings → Functions
2. Scroll to the Fluid Compute section and enable the toggle
3. Redeploy your project

This will improve concurrency handling and reduce cold start times.

## Troubleshooting

If you encounter issues with your deployment:

1. Check Vercel's deployment logs for errors
2. Verify that all required environment variables are set
3. Ensure that your code works locally before deploying
4. Check for any module imports that might not be compatible with Vercel's environment
