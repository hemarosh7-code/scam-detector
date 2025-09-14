#!/bin/bash
# ssl_setup.sh

echo "Setting up SSL certificates..."

# Install Certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Generate SSL certificate
certbot --nginx -d api.scamdetector.com --email admin@scamdetector.com --agree-tos --no-eff-email

# Generate strong DH parameters
openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -

# Test SSL configuration
echo "Testing SSL configuration..."
curl -I https://api.scamdetector.com

echo "SSL setup completed"