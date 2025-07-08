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

```bash
NEWS_API_KEY=your_newsapi_key_here
LLM_API_URL=your_llm_api_url_here
INTERPRETATION_MODEL=your_model_name_here
LLM_API_KEY=your_llm_api_key_here
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
