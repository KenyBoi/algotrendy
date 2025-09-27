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
        
        // Load trading system data when sections are activated
        this.setupTradingDataLoaders();
        
        console.log('✅ Dashboard initialized successfully');
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const contentSections = document.querySelectorAll('.content-section');

        navLinks.forEach(link => {
            link.addEventListener('click', async (e) => {
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
                    
                    // Load data for AI Systems sections
                    switch(targetId) {
                        case 'ml-models':
                            await this.loadMLModels();
                            break;
                        case 'strategies':
                            await this.loadStrategies();
                            break;
                        case 'backtesting':
                            await this.loadBacktests();
                            break;
                    }
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
                // Handle backend control button separately
                if (button.id === 'backend-control-btn') {
                    return; // Handled by setupBackendControl
                }
                
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
        
        // Setup backend control button
        this.setupBackendControl();
    }

    setupBackendControl() {
        const backendBtn = document.getElementById('backend-control-btn');
        const backendText = document.getElementById('backend-control-text');
        
        if (!backendBtn || !backendText) return;
        
        // Update button state based on backend status
        this.updateBackendControlButton();
        
        backendBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            
            const isOffline = this.backendStatus === 'backend_unreachable';
            
            if (isOffline) {
                await this.startBackend();
            } else {
                await this.stopBackend();
            }
        });
    }

    updateBackendControlButton() {
        const backendBtn = document.getElementById('backend-control-btn');
        const backendStatus = document.getElementById('backend-status');
        
        if (!backendBtn || !backendStatus) return;
        
        const isOffline = this.backendStatus === 'backend_unreachable';
        
        if (isOffline) {
            backendStatus.textContent = 'Offline - Click to Start';
            backendStatus.style.color = 'var(--danger-color)';
        } else {
            backendStatus.textContent = 'Online - Click to Stop';
            backendStatus.style.color = 'var(--success-color)';
        }
    }

    async startBackend() {
        const backendBtn = document.getElementById('backend-control-btn');
        const backendStatus = document.getElementById('backend-status');
        
        // Show loading state
        backendStatus.textContent = 'Starting...';
        backendStatus.style.color = 'var(--warning-color)';
        backendBtn.style.pointerEvents = 'none';
        
        try {
            const response = await fetch('/api/backend/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success' || data.status === 'timeout') {
                this.showMessage(data.message, data.status === 'timeout' ? 'warning' : 'success');
                
                // Wait a moment then check status
                setTimeout(async () => {
                    await this.checkBackendStatus();
                    this.updateBackendControlButton();
                }, 3000);
            } else {
                this.showMessage(data.message, 'error');
            }
        } catch (error) {
            console.error('Failed to start backend:', error);
            this.showMessage('Failed to start backend', 'error');
        } finally {
            backendBtn.style.pointerEvents = 'auto';
            if (backendStatus.textContent === 'Starting...') {
                backendStatus.textContent = 'Offline - Click to Start';
                backendStatus.style.color = 'var(--danger-color)';
            }
        }
    }

    async stopBackend() {
        const backendBtn = document.getElementById('backend-control-btn');
        const backendStatus = document.getElementById('backend-status');
        
        // Show loading state
        backendStatus.textContent = 'Stopping...';
        backendStatus.style.color = 'var(--warning-color)';
        backendBtn.style.pointerEvents = 'none';
        
        try {
            const response = await fetch('/api/backend/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showMessage(data.message, 'success');
                
                // Update status immediately
                this.backendStatus = 'backend_unreachable';
                this.updateBackendControlButton();
                this.updateBackendStatusDisplay();
            } else {
                this.showMessage(data.message, 'error');
            }
        } catch (error) {
            console.error('Failed to stop backend:', error);
            this.showMessage('Failed to stop backend', 'error');
        } finally {
            backendBtn.style.pointerEvents = 'auto';
            if (backendStatus.textContent === 'Stopping...') {
                backendStatus.textContent = 'Online - Click to Stop';
                backendStatus.style.color = 'var(--success-color)';
            }
        }
    }

    setupTradingDataLoaders() {
        // Load data when sections are first accessed
        this.dataLoaded = {
            models: false,
            strategies: false,
            backtests: false
        };
    }

    async loadMLModels() {
        if (this.dataLoaded.models) return;
        
        console.log('📊 Loading ML models...');
        const loadingEl = document.getElementById('models-loading');
        const contentEl = document.getElementById('models-content');
        
        try {
            const response = await fetch('/api/trading/models');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.renderMLModels(data.models);
            this.dataLoaded.models = true;
            
            loadingEl.style.display = 'none';
            contentEl.style.display = 'grid';
            
        } catch (error) {
            console.error('Failed to load ML models:', error);
            loadingEl.innerHTML = `<div class="error-message">Failed to load ML models: ${error.message}</div>`;
        }
    }

    async loadStrategies() {
        if (this.dataLoaded.strategies) return;
        
        console.log('⚔️ Loading trading strategies...');
        const loadingEl = document.getElementById('strategies-loading');
        const contentEl = document.getElementById('strategies-content');
        
        try {
            const response = await fetch('/api/trading/strategies');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.renderStrategies(data.strategies);
            this.dataLoaded.strategies = true;
            
            loadingEl.style.display = 'none';
            contentEl.style.display = 'grid';
            
        } catch (error) {
            console.error('Failed to load strategies:', error);
            loadingEl.innerHTML = `<div class="error-message">Failed to load strategies: ${error.message}</div>`;
        }
    }

    async loadBacktests() {
        if (this.dataLoaded.backtests) return;
        
        console.log('📈 Loading backtest results...');
        const loadingEl = document.getElementById('backtests-loading');
        const contentEl = document.getElementById('backtests-content');
        
        try {
            const response = await fetch('/api/trading/backtests');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.renderBacktests(data.backtests);
            this.dataLoaded.backtests = true;
            
            loadingEl.style.display = 'none';
            contentEl.style.display = 'block';
            
        } catch (error) {
            console.error('Failed to load backtests:', error);
            loadingEl.innerHTML = `<div class="error-message">Failed to load backtests: ${error.message}</div>`;
        }
    }

    renderMLModels(models) {
        const container = document.getElementById('models-content');
        container.innerHTML = '';
        
        models.forEach(model => {
            const modelCard = document.createElement('div');
            modelCard.className = 'model-card';
            
            modelCard.innerHTML = `
                <div class="card-header">
                    <div>
                        <div class="card-title">${model.name}</div>
                        <div class="card-subtitle">${model.symbol} • ${model.asset_type}</div>
                    </div>
                    <span class="status-badge ${model.status}">${model.status}</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">${(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Precision</span>
                        <span class="metric-value">${(model.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Recall</span>
                        <span class="metric-value">${(model.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value positive">${model.sharpe_ratio.toFixed(2)}</span>
                    </div>
                </div>
                <div style="margin-top: 1rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Last trained: ${new Date(model.last_trained).toLocaleDateString()}
                </div>
            `;
            
            container.appendChild(modelCard);
        });
    }

    renderStrategies(strategies) {
        const container = document.getElementById('strategies-content');
        container.innerHTML = '';
        
        strategies.forEach(strategy => {
            const strategyCard = document.createElement('div');
            strategyCard.className = 'strategy-card';
            
            const metrics = strategy.performance_metrics;
            
            strategyCard.innerHTML = `
                <div class="card-header">
                    <div>
                        <div class="card-title">${strategy.name}</div>
                        <div class="card-subtitle">${strategy.strategy_type} • ${strategy.asset_type}</div>
                    </div>
                    <span class="status-badge ${strategy.status}">${strategy.status}</span>
                </div>
                <div style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
                    ${strategy.description}
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">${(metrics.win_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Avg Return</span>
                        <span class="metric-value positive">${(metrics.avg_return * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Max Drawdown</span>
                        <span class="metric-value negative">-${(metrics.max_drawdown * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value positive">${metrics.sharpe_ratio.toFixed(2)}</span>
                    </div>
                </div>
            `;
            
            container.appendChild(strategyCard);
        });
    }

    renderBacktests(backtests) {
        const container = document.querySelector('.backtests-grid');
        container.innerHTML = '';
        
        backtests.forEach(backtest => {
            const backtestCard = document.createElement('div');
            backtestCard.className = 'backtest-card';
            
            backtestCard.innerHTML = `
                <div class="card-header">
                    <div>
                        <div class="card-title">${backtest.strategy_name}</div>
                        <div class="card-subtitle">${backtest.symbol} • ${backtest.start_date} to ${backtest.end_date}</div>
                    </div>
                    <span class="status-badge ${backtest.status}">${backtest.status}</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Total Return</span>
                        <span class="metric-value positive">${(backtest.total_return * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value positive">${backtest.sharpe_ratio.toFixed(2)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Max Drawdown</span>
                        <span class="metric-value negative">-${(backtest.max_drawdown * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">${(backtest.win_rate * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; color: var(--text-secondary);">
                        <span>Final Value: ${TradingUtils.formatCurrency(backtest.final_value)}</span>
                        <span>Total Trades: ${backtest.total_trades}</span>
                    </div>
                </div>
            `;
            
            container.appendChild(backtestCard);
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