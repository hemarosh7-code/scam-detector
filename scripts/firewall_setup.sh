#!/bin/bash
# firewall_setup.sh

echo "Setting up firewall rules..."

# Enable UFW
ufw --force enable

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (change port if needed)
ufw allow 22/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow database connections (only from app servers)
ufw allow from 10.0.0.0/8 to any port 5432

# Allow Redis connections (only from app servers)
ufw allow from 10.0.0.0/8 to any port 6379

# Rate limiting for HTTP/HTTPS
ufw limit 80/tcp
ufw limit 443/tcp

# Log denied connections
ufw logging on

# Block common attack ports
ufw deny 23/tcp  # Telnet
ufw deny 135/tcp # RPC
ufw deny 445/tcp # SMB

# Allow monitoring
ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus

echo "Firewall configuration completed"

# Additional iptables rules for advanced protection
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Save iptables rules
iptables-save > /etc/iptables/rules.v4