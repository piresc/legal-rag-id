#!/bin/bash

# Indonesian Legal RAG System Deployment Script
# This script deploys the AI Agent system

set -e  # Exit on any error

# Configuration
ENVIRONMENT=${ENVIRONMENT:-development}
PROJECT_NAME="indonesian-legal-rag"
AI_AGENT_DIR="ai-agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are installed"
}

# Function to create environment files
create_env_files() {
    log_info "Creating environment files..."

    # Create .env file for AI agent
    if [ ! -f "${AI_AGENT_DIR}/.env" ]; then
        cat > "${AI_AGENT_DIR}/.env" << EOF
# AI Agent Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
LOG_LEVEL=INFO

# Redis Configuration
REDIS_PASSWORD=redis_password_$(date +%s)

# LLM Configuration
DEFAULT_PROVIDER=deepseek
DEFAULT_MODEL=deepseek-reasoner
MAX_TOKENS=1000
TEMPERATURE=0.7
TIMEOUT=30

# Cache Configuration
CACHE_TTL=3600
TOP_K_DEFAULT=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Authentication
JWT_SECRET_KEY=jwt_secret_key_$(date +%s | sha256sum | cut -d' ' -f1)
ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF
        log_success "Created ${AI_AGENT_DIR}/.env"
    else
        log_warning "${AI_AGENT_DIR}/.env already exists"
    fi
    
    # Create main .env file
    if [ ! -f ".env" ]; then
        cat > ".env" << EOF
# Main Environment Configuration
ENVIRONMENT=${ENVIRONMENT}

# API Keys (set these for production)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# System Configuration
LOG_LEVEL=INFO
DEBUG=false
EOF
        log_success "Created main .env file"
    else
        log_warning "Main .env file already exists"
    fi
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."

    directories=(
        "data/raw"
        "data/processed"
        "data/vectors"
        "data/backups"
        "data/metadata"
        "ai-agent/data/raw"
        "ai-agent/data/processed"
        "ai-agent/data/vectors"
        "config"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        fi
    done
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."

    # Build AI agent images
    log_info "Building AI agent services..."
    cd "$AI_AGENT_DIR"
    docker-compose build --parallel
    cd ..

    log_success "All Docker images built successfully"
}

# Function to deploy AI agent
deploy_ai_agent() {
    log_info "Deploying AI agent services..."
    
    cd "$AI_AGENT_DIR"
    
    # Start core services first
    log_info "Starting core AI services..."
    docker-compose up -d vector-db cache-service
    
    # Wait for core services to be ready
    log_info "Waiting for core services to be ready..."
    sleep 20
    
    # Start other services
    log_info "Starting AI services..."
    docker-compose up -d
    
    cd ..
    log_success "AI agent deployed successfully"
}

# Function to deploy monitoring
deploy_monitoring() {
    log_info "Monitoring services are handled in docker-compose.yml"
    log_info "They can be started with: cd ai-agent && docker-compose up -d prometheus grafana jaeger"
}

# Function to run health checks
health_check() {
    log_info "Running health checks..."

    # Wait for services to start
    sleep 30

    # Check AI agent services
    log_info "Checking AI agent services..."
    cd "$AI_AGENT_DIR"

    services=("vector-db" "cache-service" "rag-service" "llm-service" "api-gateway")
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            log_success "✓ $service is running"
        else
            log_error "✗ $service is not running"
        fi
    done
    cd ..

    log_success "Health checks completed"
}

# Function to show service URLs
show_urls() {
    log_info "Service URLs:"
    echo ""
    echo "AI Agent Services:"
    echo "  - API Gateway: http://localhost:8000"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Jaeger: http://localhost:16686"
}

# Function to stop all services
stop_services() {
    log_info "Stopping all services..."

    cd "$AI_AGENT_DIR"
    docker-compose down
    cd ..

    log_success "All services stopped"
}

# Function to clean up
cleanup() {
    log_info "Cleaning up..."

    # Stop services
    stop_services

    # Remove containers
    cd "$AI_AGENT_DIR"
    docker-compose down -v --remove-orphans
    cd ..

    # Remove unused images and volumes
    docker system prune -f
    docker volume prune -f

    log_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    local service=$1

    if [ -z "$service" ]; then
        log_info "Available services:"
        echo "AI Agent: rag, llm, api-gateway, vector-db, cache"
        echo "Usage: $0 logs <service-name>"
        return
    fi

    case "$service" in
        "rag"|"llm"|"api-gateway"|"vector-db"|"cache")
            cd "$AI_AGENT_DIR"
            docker-compose logs -f "$service-service"
            cd ..
            ;;
        *)
            log_error "Unknown service: $service"
            ;;
    esac
}

# Function to backup data
backup_data() {
    log_info "Creating backup..."
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup data directories
    cp -r data "$backup_dir/"
    
    # Backup Docker volumes
    docker run --rm -v $(pwd)/data:/source -v $(pwd)/"$backup_dir":/backup alpine tar czf /backup/data_backup.tar.gz -C /source .
    
    log_success "Backup created in $backup_dir"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of Indonesian Legal RAG System..."
    log_info "Environment: $ENVIRONMENT"

    check_docker
    create_env_files
    create_directories
    build_images
    deploy_ai_agent
    deploy_monitoring
    health_check
    show_urls

    log_success "Deployment completed successfully!"
    log_info "Run './deploy.sh logs <service>' to view service logs"
    log_info "Run './deploy.sh stop' to stop all services"
    log_info "Run './deploy.sh cleanup' to clean up everything"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 5
        deploy
        ;;
    "logs")
        show_logs "$2"
        ;;
    "health")
        health_check
        ;;
    "backup")
        backup_data
        ;;
    "cleanup")
        cleanup
        ;;
    "urls")
        show_urls
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  deploy     Deploy the AI Agent system (default)"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  logs       Show logs for a service"
        echo "  health     Run health checks"
        echo "  backup     Create backup of data"
        echo "  cleanup    Stop services and clean up"
        echo "  urls       Show service URLs"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 deploy              # Deploy AI Agent system"
        echo "  $0 logs rag            # Show RAG service logs"
        echo "  $0 stop                # Stop all services"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac