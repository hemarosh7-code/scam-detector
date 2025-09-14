# deploy.sh - Google Cloud deployment script
#!/bin/bash

echo "Starting Google Cloud deployment..."

# Set project and region
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="scam-detector"

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com

# Create Cloud SQL instance
gcloud sql instances create scam-detector-db \
    --database-version=POSTGRES_13 \
    --tier=db-f1-micro \
    --region=$REGION

# Create database
gcloud sql databases create scam_detector --instance=scam-detector-db

# Create user
gcloud sql users create scam_user \
    --instance=scam-detector-db \
    --password=your-secure-password

# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars DATABASE_URL="postgresql://scam_user:your-secure-password@/scam_detector?host=/cloudsql/$PROJECT_ID:$REGION:scam-detector-db" \
    --add-cloudsql-instances $PROJECT_ID:$REGION:scam-detector-db

echo "Deployment completed!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')"