#!/bin/bash
# automated_deployment.sh

set -e  # Exit on any error

PROJECT_NAME="scam-detector"
VERSION=$(date +"%Y%m%d-%H%M%S")
BACKUP_DIR="/backup/scam-detector"

echo "Starting automated deployment v$VERSION..."

# Create backup
echo "Creating backup..."
mkdir -p $BACKUP_DIR/$VERSION
cp -r /app $BACKUP_DIR/$VERSION/
cp /etc/nginx/sites-available/$PROJECT_NAME $BACKUP_DIR/$VERSION/

# Pull latest code
echo "Pulling latest code..."
cd /app
git pull origin main

# Run tests
echo "Running tests..."
python -m pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed! Aborting deployment."
    exit 1
fi

# Build new Docker image
echo "Building Docker image..."
docker build -t $PROJECT_NAME:$VERSION .
docker tag $PROJECT_NAME:$VERSION $PROJECT_NAME:latest

# Database migrations (if needed)
echo "Running database migrations..."
docker-compose exec scam-detector python manage.py db upgrade

# Rolling deployment
echo "Performing rolling deployment..."
docker-compose up -d --scale scam-detector=2
sleep 30

# Health check
echo "Performing health check..."
for i in {1..5}; do
    if curl -f http://localhost/health; then
        echo "Health check passed"
        break
    else
        if [ $i -eq 5 ]; then
            echo "Health check failed! Rolling back..."
            docker-compose up -d --scale scam-detector=1
            docker run --rm -v /backup/$VERSION:/backup $PROJECT_NAME:$VERSION cp -r /backup/app /
            exit 1
        fi
        sleep 10
    fi
done

# Scale down old instances
docker-compose up -d --scale scam-detector=1

# Cleanup old images (keep last 5)
echo "Cleaning up old Docker images..."
docker images $PROJECT_NAME --format "table {{.ID}}\t{{.Tag}}" | tail -n +6 | awk '{print $1}' | xargs -r docker rmi

# Update monitoring alerts
echo "Updating monitoring configuration..."
curl -X POST http://prometheus:9090/-/reload

echo "Deployment completed successfully!"

# Send notification
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"Scam Detector v$VERSION deployed successfully\"}" \
    $SLACK_WEBHOOK_URL