# Infrastructure Components

This document explains the optional infrastructure components that support the Indonesian Legal RAG System in production environments: **Nginx**, **Consul**, and **RabbitMQ**.

## Overview

| Component | Purpose | When to Use | Profile |
|-----------|---------|-------------|---------|
| **Nginx** | Load balancer & reverse proxy | Production deployments | `production` |
| **Consul** | Service discovery & health monitoring | Distributed deployments | `discovery` |
| **RabbitMQ** | Message queue for async processing | Background job processing | `queue` |

These components are **optional** and can be enabled via Docker Compose profiles when needed.

## Nginx

### What is Nginx?

Nginx is a high-performance web server and reverse proxy that handles:

- **Load Balancing**: Distribute traffic across multiple service instances
- **SSL Termination**: Handle HTTPS encryption/decryption
- **Static File Serving**: Serve frontend assets efficiently
- **Request Routing**: Route requests to different backend services
- **Rate Limiting**: Control request rates at the edge
- **Caching**: Cache responses to reduce backend load

### Why Do You Need Nginx?

#### 1. **Load Balancing**

Without Nginx:
```
Client → API Gateway (Single Point of Failure)
```

With Nginx:
```
            ┌─────────────┐
            │   Nginx     │
            │ Load Balancer│
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
  Gateway 1   Gateway 2   Gateway 3
```

**Benefits**:
- No single point of failure
- Horizontal scaling
- Better resource utilization

#### 2. **SSL/TLS Termination**

Nginx handles HTTPS encryption, so your backend services don't have to:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://api-gateway:8000;
    }
}
```

**Benefits**:
- Simplified backend configuration
- Centralized certificate management
- Better performance (SSL optimization)

#### 3. **Static File Serving**

If you have a web frontend, Nginx serves it efficiently:

```nginx
location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}

location /api {
    proxy_pass http://api-gateway:8000;
}
```

**Benefits**:
- Fast static file delivery
- Reduced load on application servers
- Better caching strategies

#### 4. **Security Layer**

Nginx provides an additional security layer:

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20;
    proxy_pass http://api-gateway:8000;
}
```

**Benefits**:
- DDoS protection
- Request rate limiting
- IP whitelisting/blacklisting

### Configuration

#### Docker Compose

```yaml
nginx:
  image: nginx:alpine
  container_name: legal-rag-nginx
  volumes:
    - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    - ./nginx/ssl:/etc/nginx/ssl
  ports:
    - "80:80"
    - "443:443"
  depends_on:
    - api-gateway
  networks:
    - legal-rag-ai-network
  restart: unless-stopped
  profiles:
    - production  # Only start with --profile production
```

#### Example nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    # Upstream for load balancing
    upstream api_gateway {
        least_conn;  # Load balancing strategy
        server api-gateway:8000;
        # Add more instances for scaling:
        # server api-gateway-2:8000;
        # server api-gateway-3:8000;
    }

    # Rate limiting zone
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name api.example.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.example.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        # API endpoint
        location /api {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://api_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://api_gateway/health;
            access_log off;
        }
    }
}
```

### Usage

#### Enable Nginx

```bash
# Start with Nginx enabled
docker-compose --profile production up -d

# Or add to existing deployment
docker-compose --profile production up -d nginx
```

#### SSL Certificate Setup

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Generate self-signed certificate (for development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/CN=api.example.com"

# For production, use Let's Encrypt or your organization's CA
```

### When to Use Nginx

**Use Nginx if you need**:
- ✅ Production deployment with multiple API Gateway instances
- ✅ HTTPS/SSL support
- ✅ Static file serving (web frontend)
- ✅ Advanced load balancing
- ✅ Edge rate limiting and security

**Skip Nginx if**:
- ❌ Development environment
- ❌ Single-instance deployment
- ❌ No SSL requirement
- ❌ Simple deployment

---

## Consul

### What is Consul?

Consul is a service networking solution that provides:

- **Service Discovery**: Automatically find and connect services
- **Health Checking**: Monitor service health
- **KV Store**: Distributed configuration storage
- **Service Mesh**: Secure service-to-service communication

### Why Do You Need Consul?

#### 1. **Dynamic Service Discovery**

Without Consul:
```python
# Hardcoded service URLs
RAG_SERVICE_URL = "http://rag-service:8001"  # What if it moves?
```

With Consul:
```python
# Dynamic service discovery
import consul

c = consul.Consul()
_, services = c.health.service('rag-service', passing=True)
rag_service_url = f"http://{services[0]['Service']['Address']}:{services[0]['Service']['Port']}"
```

**Benefits**:
- Services can move without breaking clients
- Automatic failover to healthy instances
- Dynamic scaling without configuration changes

#### 2. **Health Monitoring**

Consul continuously checks service health:

```python
# Service registers with health check
consul.agent.service.register(
    name='rag-service',
    service_id='rag-service-1',
    address='rag-service',
    port=8001,
    check=consul.Check.http('http://rag-service:8001/health', interval='10s')
)
```

**Benefits**:
- Automatic detection of failed services
- Traffic routing only to healthy instances
- Alerting on service failures

#### 3. **Load Balancing**

Consul provides all healthy instances:

```python
# Get all healthy instances
_, services = consul.health.service('rag-service', passing=True)

# Round-robin or random selection
instance = random.choice(services)
target = f"http://{instance['Service']['Address']}:{instance['Service']['Port']}"
```

**Benefits**:
- Client-side load balancing
- Better resource utilization
- Improved fault tolerance

#### 4. **Configuration Management**

Use Consul KV for distributed configuration:

```bash
# Store configuration
consul kv put config/rag_service/max_tokens "1000"
consul kv put config/rag_service/temperature "0.7"

# Retrieve in application
max_tokens = consul.kv.get('config/rag_service/max_tokens')[1]['Value']
```

**Benefits**:
- Centralized configuration
- Dynamic configuration updates
- Environment-specific settings

### Configuration

#### Docker Compose

```yaml
consul:
  image: consul:latest
  container_name: legal-rag-consul
  command: agent -server -bootstrap -ui -client=0.0.0.0
  volumes:
    - consul_data:/consul/data
  ports:
    - "8500:8500"    # HTTP API
    - "8600:8600/udp" # DNS interface
  networks:
    - legal-rag-ai-network
  restart: unless-stopped
  profiles:
    - discovery  # Only start with --profile discovery
```

### Service Registration

#### Register Service with Consul

```python
import consul

class ConsulServiceRegistry:
    def __init__(self, consul_host='localhost', consul_port=8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)

    def register(self, service_name, service_id, address, port, health_check_url):
        """Register service with Consul"""
        self.consul.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=address,
            port=port,
            check=consul.Check.http(health_check_url, interval='10s'),
            tags=['legal-rag', 'ai']
        )

    def deregister(self, service_id):
        """Deregister service"""
        self.consul.agent.service.deregister(service_id)

    def discover(self, service_name):
        """Discover healthy service instances"""
        _, services = self.consul.health.service(service_name, passing=True)
        return [
            {
                'id': s['Service']['ID'],
                'address': s['Service']['Address'],
                'port': s['Service']['Port']
            }
            for s in services
        ]

# Usage
registry = ConsulServiceRegistry()
registry.register(
    service_name='rag-service',
    service_id='rag-service-1',
    address='rag-service',
    port=8001,
    health_check_url='http://rag-service:8001/health'
)
```

### Consul UI

Access the Consul web UI:

```
http://localhost:8500
```

Features:
- View all registered services
- Check service health status
- Monitor service instances
- Access KV store
- View service topology

### When to Use Consul

**Use Consul if you need**:
- ✅ Dynamic service discovery (multiple instances)
- ✅ Automatic failover and load balancing
- ✅ Distributed configuration management
- ✅ Health monitoring and alerting
- ✅ Service mesh for secure communication

**Skip Consul if**:
- ❌ Development environment
- ❌ Single-instance deployment
- ❌ Static service configuration is sufficient
- ❌ Simple deployment

---

## RabbitMQ

### What is RabbitMQ?

RabbitMQ is a message broker that implements:

- **Message Queuing**: Asynchronous task processing
- **Pub/Sub**: Broadcast messages to multiple consumers
- **Routing**: Direct messages to specific queues
- **Reliability**: Message persistence and acknowledgment

### Why Do You Need RabbitMQ?

#### 1. **Asynchronous Processing**

Without RabbitMQ:
```python
# Synchronous - blocks response
response = process_heavy_task(data)  # Takes 10 seconds
return response  # User waits 10 seconds
```

With RabbitMQ:
```python
# Asynchronous - immediate response
publish_task(data)  # Queues task, returns immediately
return {"status": "processing", "task_id": "123"}  # User gets instant response

# Background worker processes task
def worker():
    while True:
        task = get_task_from_queue()
        result = process_heavy_task(task.data)
        notify_user(task.user_id, result)
```

**Benefits**:
- Faster user response times
- Better resource utilization
- Resilience to peak loads

#### 2. **Task Queuing**

Queue heavy tasks for background processing:

```python
# Queue document indexing task
message = {
    'task': 'index_document',
    'document_id': doc_id,
    'priority': 'high'
}

channel.basic_publish(
    exchange='',
    routing_key='document_tasks',
    body=json.dumps(message)
)
```

**Benefits**:
- Process tasks in order
- Priority queues
- Retry failed tasks
- Task scheduling

#### 3. **Event-Driven Architecture**

Pub/sub for event broadcasting:

```python
# Publisher: Document updated
channel.basic_publish(
    exchange='document_events',
    routing_key='document.updated',
    body=json.dumps({'doc_id': doc_id, 'timestamp': now()})
)

# Subscribers receive events
# - Search index updater
# - Cache invalidator
# - Analytics collector
# - Notification service
```

**Benefits**:
- Loose coupling between services
- Event-driven updates
- Multiple independent consumers
- Scalable processing

#### 4. **Load Distribution**

Distribute work across multiple workers:

```
┌──────────────┐
│   RabbitMQ   │
│ Task Queue   │
└──────┬───────┘
       │
       ├──────► Worker 1 (Processing)
       ├──────► Worker 2 (Processing)
       ├──────► Worker 3 (Processing)
       └──────► Worker 4 (Processing)
```

**Benefits**:
- Parallel processing
- Automatic load balancing
- Worker scaling
- Fault tolerance

### Configuration

#### Docker Compose

```yaml
rabbitmq:
  image: rabbitmq:3-management-alpine
  container_name: legal-rag-rabbitmq
  environment:
    - RABBITMQ_DEFAULT_USER=admin
    - RABBITMQ_DEFAULT_PASS=admin
  volumes:
    - rabbitmq_data:/var/lib/rabbitmq
  ports:
    - "5672:5672"   # AMQP protocol
    - "15672:15672" # Management UI
  networks:
    - legal-rag-ai-network
  restart: unless-stopped
  profiles:
    - queue  # Only start with --profile queue
```

### Usage Examples

#### Publishing Tasks

```python
import pika
import json

class TaskPublisher:
    def __init__(self, rabbitmq_url='amqp://admin:admin@localhost:5672/'):
        connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        self.channel = connection.channel()

        # Declare queue
        self.channel.queue_declare(queue='document_tasks', durable=True)

    def publish_index_task(self, document_id: str, priority: str = 'normal'):
        """Publish document indexing task"""
        message = {
            'task': 'index_document',
            'document_id': document_id,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        }

        self.channel.basic_publish(
            exchange='',
            routing_key='document_tasks',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )

# Usage
publisher = TaskPublisher()
publisher.publish_index_task('doc_12345', priority='high')
```

#### Consuming Tasks

```python
import pika
import json

class TaskWorker:
    def __init__(self, rabbitmq_url='amqp://admin:admin@localhost:5672/'):
        connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        self.channel = connection.channel()
        self.channel.queue_declare(queue='document_tasks', durable=True)

    def start(self):
        """Start consuming tasks"""
        self.channel.basic_consume(
            queue='document_tasks',
            on_message_callback=self.process_task,
            auto_ack=False
        )

        print('Worker started, waiting for tasks...')
        self.channel.start_consuming()

    def process_task(self, ch, method, properties, body):
        """Process task"""
        try:
            task = json.loads(body)

            if task['task'] == 'index_document':
                self.index_document(task['document_id'])

            # Acknowledge successful processing
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error: {e}")
            # Reject and requeue task
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def index_document(self, document_id):
        """Index document (example task)"""
        print(f"Indexing document {document_id}...")
        # Do actual work here
        time.sleep(5)  # Simulate work
        print(f"Document {document_id} indexed!")

# Usage
worker = TaskWorker()
worker.start()
```

### RabbitMQ Management UI

Access the RabbitMQ management interface:

```
URL: http://localhost:15672
Username: admin
Password: admin
```

Features:
- View queues and messages
- Monitor message rates
- Manage connections and channels
- Create/delete queues and exchanges
- Monitor consumer activity

### Use Cases for Legal RAG System

1. **Document Indexing**
   - Queue new documents for indexing
   - Process large documents asynchronously
   - Retry failed indexing attempts

2. **Cache Invalidation**
   - Publish document update events
   - Invalidate related cache entries
   - Update search indices

3. **Batch Processing**
   - Nightly document reindexing
   - Bulk data imports
   - Periodic data cleanup

4. **Notifications**
   - Email alerts for system events
   - WebSocket push updates
   - Analytics and logging

### When to Use RabbitMQ

**Use RabbitMQ if you need**:
- ✅ Asynchronous task processing
- ✅ Background job queues
- ✅ Event-driven architecture
- ✅ Message persistence and reliability
- ✅ Work distribution across workers
- ✅ Long-running tasks

**Skip RabbitMQ if**:
- ❌ All operations are fast (<1 second)
- ❌ Simple request/response is sufficient
- ❌ No background processing needed
- ❌ Development/testing environment

---

## Comparison

| Component | Primary Use Case | Complexity | Essential for Production? |
|-----------|------------------|------------|---------------------------|
| **Nginx** | Load balancing & SSL | Medium | Recommended |
| **Consul** | Service discovery | High | Optional (for scaling) |
| **RabbitMQ** | Async processing | Medium | Optional (for background tasks) |

## Quick Start

### Enable All Components

```bash
# Start with all infrastructure components
docker-compose --profile production --profile discovery --profile queue up -d

# Check status
docker-compose ps

# Access management UIs
# Consul: http://localhost:8500
# RabbitMQ: http://localhost:15672
```

### Enable Specific Components

```bash
# Nginx only (production)
docker-compose --profile production up -d nginx

# Consul only (service discovery)
docker-compose --profile discovery up -d consul

# RabbitMQ only (async tasks)
docker-compose --profile queue up -d rabbitmq

# Nginx + RabbitMQ
docker-compose --profile production --profile queue up -d
```

## Best Practices

### Nginx
- ✅ Use upstream for load balancing
- ✅ Enable SSL in production
- ✅ Implement rate limiting
- ✅ Monitor access logs
- ✅ Keep configuration version-controlled

### Consul
- ✅ Use health checks for all services
- ✅ Implement service discovery clients
- ✅ Monitor Consul UI for issues
- ✅ Backup Consul data regularly
- ✅ Use ACLs for security in production

### RabbitMQ
- ✅ Use durable queues for important tasks
- ✅ Implement proper error handling
- ✅ Monitor queue depths
- ✅ Set up alerting for stuck messages
- ✅ Use message acknowledgments correctly

## References

- **Nginx Docs**: https://nginx.org/en/docs/
- **Consul Docs**: https://developer.hashicorp.com/consul/docs
- **RabbitMQ Docs**: https://www.rabbitmq.com/docs
- **Docker Compose**: [`docker-compose.yml`](../ai-agent/docker-compose.yml)
