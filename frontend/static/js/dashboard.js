/**
 * AlgoTrendy Dashboard JavaScript
 * Modular frontend controller for trading platform
 */

class TradingDashboard {
    constructor() {
        this.backendUrl = window.location.protocol === 'https:' 
            ? window.location.origin.replace(':5000', ':8000')
            : 'http://localhost:8000';
        this.connectionStatus = document.getElementById('connection-status');
        this.backendStatusElement = document.getElementById('backend-status');
        this.apiEndpointElement = document.getElementById('api-endpoint');
        
        this.init();
    }

    async init() {
        console.log('🚀 AlgoTrendy Dashboard initializing...');
        
        // Set up navigation
        this.setupNavigation();
        
        // Check backend connection
        await this.checkBackendStatus();
        
        // Start periodic status checks
        this.startStatusMonitoring();
        
        // Setup action buttons
        this.setupActionButtons();
        
        console.log('✅ Dashboard initialized successfully');
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const contentSections = document.querySelectorAll('.content-section');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                const targetId = link.getAttribute('href').substring(1);
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                // Show target content section
                contentSections.forEach(section => {
                    section.classList.remove('active');
                });
                
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add('active');
                }
            });
        });
    }

    async checkBackendStatus() {
        try {
            console.log(`🔍 Checking backend status at: ${this.backendUrl}`);
            
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'backend_unreachable') {
                this.updateConnectionStatus('disconnected', 'Backend Offline');
                this.backendStatusElement.textContent = 'Offline';
                this.backendStatusElement.style.color = 'var(--danger-color)';
            } else {
                this.updateConnectionStatus('connected', 'Connected');
                this.backendStatusElement.textContent = 'Online';
                this.backendStatusElement.style.color = 'var(--success-color)';
            }
            
            this.apiEndpointElement.textContent = this.backendUrl;
            
        } catch (error) {
            console.error('❌ Backend status check failed:', error);
            this.updateConnectionStatus('disconnected', 'Connection Error');
            this.backendStatusElement.textContent = 'Error';
            this.backendStatusElement.style.color = 'var(--danger-color)';
            this.apiEndpointElement.textContent = 'Unreachable';
        }
    }

    updateConnectionStatus(status, text) {
        const statusBadge = this.connectionStatus;
        
        // Remove existing status classes
        statusBadge.classList.remove('connected', 'disconnected');
        statusBadge.classList.add(status);
        
        // Update text
        const statusText = statusBadge.querySelector('span');
        if (statusText) {
            statusText.textContent = text;
        }
    }

    startStatusMonitoring() {
        // Check backend status every 30 seconds
        setInterval(() => {
            this.checkBackendStatus();
        }, 30000);
    }

    setupActionButtons() {
        const actionButtons = document.querySelectorAll('.action-btn');
        
        actionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const buttonText = button.textContent.trim();
                
                switch(buttonText) {
                    case 'Start Trading Interface':
                        this.showMessage('Trading interface would launch here...', 'info');
                        break;
                    case 'View Backtests':
                        this.showMessage('Backtest results would display here...', 'info');
                        break;
                    case 'System Settings':
                        this.showMessage('System settings panel would open here...', 'info');
                        break;
                    default:
                        this.showMessage(`${buttonText} clicked`, 'info');
                }
            });
        });
    }

    showMessage(message, type = 'info') {
        // Simple toast notification system
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        // Style the toast
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 20px',
            borderRadius: '6px',
            color: 'white',
            fontWeight: '500',
            zIndex: '1000',
            opacity: '0',
            transform: 'translateY(-10px)',
            transition: 'all 0.3s ease'
        });
        
        // Set background color based on type
        const colors = {
            info: 'var(--primary-color)',
            success: 'var(--success-color)',
            warning: 'var(--warning-color)',
            error: 'var(--danger-color)'
        };
        toast.style.backgroundColor = colors[type] || colors.info;
        
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }

    // API Methods for future integration
    async callBackendAPI(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            const response = await fetch(`/api/proxy${endpoint}`, options);
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }
}

// CSS variables for JavaScript access
const cssVars = {
    primaryColor: '#2563eb',
    successColor: '#10b981',
    warningColor: '#f59e0b',
    dangerColor: '#ef4444'
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});

// Global utility functions
window.TradingUtils = {
    formatCurrency: (value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    },
    
    formatPercentage: (value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2
        }).format(value / 100);
    },
    
    formatNumber: (value) => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
};