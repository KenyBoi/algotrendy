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
                    
                    // Load data for AI Systems sections and Portfolio
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
                        case 'portfolio':
                            await this.loadPortfolioData();
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

    // =============================================================================
    // PORTFOLIO MANAGEMENT FUNCTIONS
    // =============================================================================

    async loadPortfolioData() {
        console.log('💰 Loading portfolio data...');
        
        // Initialize portfolio state tracking
        if (!this.portfolioLoaded) {
            this.setupPortfolioEventHandlers();
            this.portfolioLoaded = true;
        }

        // Load all portfolio data sections
        await Promise.all([
            this.loadPortfolioOverview(),
            this.loadPositions(),
            this.loadPerformanceData()
        ]);

        // Show portfolio content
        const loadingEl = document.getElementById('portfolio-loading');
        const contentEl = document.getElementById('portfolio-content');
        
        if (loadingEl && contentEl) {
            loadingEl.style.display = 'none';
            contentEl.style.display = 'block';
        }
    }

    async loadPortfolioOverview() {
        try {
            console.log('📊 Loading portfolio overview...');
            
            const response = await fetch('/api/proxy/portfolio');
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update portfolio metrics
            this.updatePortfolioMetrics(data);
            
        } catch (error) {
            console.error('Failed to load portfolio overview:', error);
            this.showPortfolioError('portfolio-overview', 'Failed to load portfolio overview');
        }
    }

    updatePortfolioMetrics(data) {
        // Update main account summary
        this.updateElement('total-portfolio-value', this.formatCurrency(data.total_value));
        this.updateElement('cash-balance', this.formatCurrency(data.cash_balance));
        this.updateElement('invested-amount', this.formatCurrency(data.invested_amount));

        // Update P&L summary with color coding
        this.updatePnLElement('unrealized-pnl', data.unrealized_pnl);
        this.updatePnLElement('realized-pnl', data.realized_pnl);
        this.updatePnLElement('day-change', data.day_change);
        this.updatePnLElement('day-change-percent', data.day_change_percent, true);

        // Update risk metrics
        this.updateElement('portfolio-beta', data.risk_metrics?.beta?.toFixed(2) || '-');
        this.updateElement('sharpe-ratio', data.risk_metrics?.sharpe_ratio?.toFixed(2) || '-');
        this.updateElement('max-drawdown', data.risk_metrics?.max_drawdown ? 
            (data.risk_metrics.max_drawdown * 100).toFixed(1) + '%' : '-');
        this.updateElement('volatility', data.risk_metrics?.volatility ? 
            (data.risk_metrics.volatility * 100).toFixed(1) + '%' : '-');

        // Update performance periods
        if (data.portfolio_performance) {
            Object.entries(data.portfolio_performance).forEach(([period, value]) => {
                const elementId = `perf-${period}`;
                const percentage = (value * 100).toFixed(2) + '%';
                this.updatePnLElement(elementId, percentage);
            });
        }

        // Update last updated timestamp
        this.updateElement('portfolio-last-updated', 
            new Date(data.last_updated).toLocaleTimeString());
    }

    async loadPositions() {
        try {
            console.log('📈 Loading positions...');
            
            const loadingEl = document.getElementById('positions-loading');
            const contentEl = document.getElementById('positions-content');
            
            if (loadingEl) loadingEl.style.display = 'block';
            if (contentEl) contentEl.style.display = 'none';

            const response = await fetch('/api/proxy/positions');
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update positions summary
            this.updateElement('positions-count', `${data.total_positions} positions`);
            this.updateElement('positions-total-value', `Total: ${this.formatCurrency(data.total_market_value)}`);

            // Populate positions table
            this.populatePositionsTable(data.positions);

            // Setup table functionality
            this.setupPositionsTable();

            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
            
        } catch (error) {
            console.error('Failed to load positions:', error);
            this.showPositionsError('Failed to load positions');
        }
    }

    populatePositionsTable(positions) {
        const tbody = document.getElementById('positions-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        positions.forEach(position => {
            const row = document.createElement('tr');
            row.className = 'position-row';
            row.dataset.symbol = position.symbol;
            row.dataset.assetType = position.asset_type;

            row.innerHTML = `
                <td class="symbol-cell">
                    <div class="symbol-info">
                        <span class="symbol">${position.symbol}</span>
                        <span class="position-type ${position.position_type}">${position.position_type.toUpperCase()}</span>
                    </div>
                </td>
                <td class="name-cell">${position.name}</td>
                <td class="asset-type-cell">
                    <span class="asset-badge ${position.asset_type}">${this.formatAssetType(position.asset_type)}</span>
                </td>
                <td class="numeric-cell">${this.formatQuantity(position.quantity)}</td>
                <td class="numeric-cell">${this.formatCurrency(position.current_price)}</td>
                <td class="numeric-cell">${this.formatCurrency(position.market_value)}</td>
                <td class="numeric-cell pnl-cell">
                    <div class="pnl-info">
                        <span class="pnl-value ${this.getPnLClass(position.unrealized_pnl)}">
                            ${this.formatCurrency(position.unrealized_pnl)}
                        </span>
                        <span class="pnl-percent ${this.getPnLClass(position.unrealized_pnl_percent)}">
                            (${(position.unrealized_pnl_percent * 100).toFixed(2)}%)
                        </span>
                    </div>
                </td>
                <td class="numeric-cell pnl-cell">
                    <div class="pnl-info">
                        <span class="pnl-value ${this.getPnLClass(position.day_change)}">
                            ${this.formatCurrency(position.day_change)}
                        </span>
                        <span class="pnl-percent ${this.getPnLClass(position.day_change_percent)}">
                            (${(position.day_change_percent * 100).toFixed(2)}%)
                        </span>
                    </div>
                </td>
                <td class="actions-cell">
                    <div class="action-buttons">
                        <button class="action-btn small" onclick="dashboard.viewPositionDetails('${position.symbol}')">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="action-btn small warning" onclick="dashboard.closePosition('${position.symbol}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    setupPositionsTable() {
        // Setup search functionality
        const searchInput = document.getElementById('positions-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterPositions(e.target.value);
            });
        }

        // Setup filter functionality
        const filterSelect = document.getElementById('positions-filter');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.filterPositionsByType(e.target.value);
            });
        }

        // Setup sorting functionality
        const sortableHeaders = document.querySelectorAll('.positions-table .sortable');
        sortableHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const sortKey = header.dataset.sort;
                this.sortPositions(sortKey);
            });
        });
    }

    async loadPerformanceData() {
        try {
            console.log('📊 Loading performance data...');
            
            const loadingEl = document.getElementById('performance-loading');
            const contentEl = document.getElementById('performance-content');
            
            if (loadingEl) loadingEl.style.display = 'block';
            if (contentEl) contentEl.style.display = 'none';

            const response = await fetch('/api/proxy/portfolio/performance');
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update performance metrics
            this.updatePerformanceMetrics(data.summary);

            // Create simple performance chart
            this.createPerformanceChart(data.performance_data);

            // Populate performance table
            this.populatePerformanceTable(data.performance_data);

            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
            
        } catch (error) {
            console.error('Failed to load performance data:', error);
            this.showPerformanceError('Failed to load performance data');
        }
    }

    updatePerformanceMetrics(summary) {
        this.updatePnLElement('total-return', summary.total_return + '%');
        this.updateElement('best-day', summary.best_day + '%');
        this.updateElement('worst-day', summary.worst_day + '%');
        this.updateElement('avg-daily-return', summary.avg_daily_return + '%');
    }

    createPerformanceChart(performanceData) {
        const canvas = document.getElementById('performance-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (!performanceData || performanceData.length === 0) return;

        // Get data values
        const values = performanceData.map(d => d.portfolio_value);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;

        // Chart dimensions
        const chartPadding = 40;
        const chartWidth = width - chartPadding * 2;
        const chartHeight = height - chartPadding * 2;

        // Draw chart background
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        // Draw chart line
        ctx.beginPath();
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;

        performanceData.forEach((point, index) => {
            const x = chartPadding + (index / (performanceData.length - 1)) * chartWidth;
            const y = chartPadding + chartHeight - ((point.portfolio_value - minValue) / valueRange) * chartHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw chart axes
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(chartPadding, chartPadding);
        ctx.lineTo(chartPadding, height - chartPadding);
        ctx.stroke();
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(chartPadding, height - chartPadding);
        ctx.lineTo(width - chartPadding, height - chartPadding);
        ctx.stroke();
    }

    populatePerformanceTable(performanceData) {
        const tbody = document.getElementById('performance-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        // Show only last 10 days
        const recentData = performanceData.slice(-10);

        recentData.forEach(point => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${new Date(point.date).toLocaleDateString()}</td>
                <td>${this.formatCurrency(point.portfolio_value)}</td>
                <td class="${this.getPnLClass(point.daily_return)}">${point.daily_return.toFixed(2)}%</td>
                <td class="${this.getPnLClass(point.cumulative_return)}">${point.cumulative_return.toFixed(2)}%</td>
            `;
            tbody.appendChild(row);
        });
    }

    setupPortfolioEventHandlers() {
        // Setup refresh button
        const refreshBtn = document.getElementById('refresh-portfolio-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadPortfolioData();
            });
        }

        // Setup chart period buttons
        const periodButtons = document.querySelectorAll('.chart-period-btn');
        periodButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                periodButtons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                // TODO: Implement period filtering
                this.showMessage(`Loading ${e.target.dataset.period}-day performance...`, 'info');
            });
        });

        // Setup auto-refresh for portfolio data
        this.startPortfolioAutoRefresh();
    }

    startPortfolioAutoRefresh() {
        // Refresh portfolio data every 30 seconds
        if (this.portfolioRefreshInterval) {
            clearInterval(this.portfolioRefreshInterval);
        }
        
        this.portfolioRefreshInterval = setInterval(() => {
            // Only refresh if portfolio tab is active
            const portfolioSection = document.getElementById('portfolio');
            if (portfolioSection && portfolioSection.classList.contains('active')) {
                this.loadPortfolioOverview();
                this.loadPositions();
            }
        }, 30000);
    }

    // Portfolio utility functions
    filterPositions(searchTerm) {
        const rows = document.querySelectorAll('.position-row');
        const term = searchTerm.toLowerCase();

        rows.forEach(row => {
            const symbol = row.dataset.symbol.toLowerCase();
            const name = row.querySelector('.name-cell').textContent.toLowerCase();
            
            if (symbol.includes(term) || name.includes(term)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    filterPositionsByType(assetType) {
        const rows = document.querySelectorAll('.position-row');
        
        rows.forEach(row => {
            if (!assetType || row.dataset.assetType === assetType) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    sortPositions(sortKey) {
        const tbody = document.getElementById('positions-table-body');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('.position-row'));
        
        rows.sort((a, b) => {
            const aValue = this.getPositionSortValue(a, sortKey);
            const bValue = this.getPositionSortValue(b, sortKey);
            
            if (typeof aValue === 'number' && typeof bValue === 'number') {
                return bValue - aValue; // Descending for numbers
            } else {
                return aValue.localeCompare(bValue); // Ascending for strings
            }
        });

        // Re-append sorted rows
        rows.forEach(row => tbody.appendChild(row));
    }

    getPositionSortValue(row, sortKey) {
        switch (sortKey) {
            case 'symbol':
                return row.dataset.symbol;
            case 'name':
                return row.querySelector('.name-cell').textContent;
            case 'asset_type':
                return row.dataset.assetType;
            case 'quantity':
                return parseFloat(row.querySelector('.numeric-cell:nth-of-type(4)').textContent);
            case 'current_price':
                return this.parseMoneyValue(row.querySelector('.numeric-cell:nth-of-type(5)').textContent);
            case 'market_value':
                return this.parseMoneyValue(row.querySelector('.numeric-cell:nth-of-type(6)').textContent);
            case 'unrealized_pnl':
                return this.parseMoneyValue(row.querySelector('.pnl-value').textContent);
            case 'day_change':
                return this.parseMoneyValue(row.querySelectorAll('.pnl-value')[1].textContent);
            default:
                return 0;
        }
    }

    // Portfolio action handlers
    viewPositionDetails(symbol) {
        this.showMessage(`Viewing details for ${symbol}`, 'info');
        // TODO: Implement position details modal/view
    }

    closePosition(symbol) {
        if (confirm(`Are you sure you want to close the position in ${symbol}?`)) {
            this.showMessage(`Closing position in ${symbol}...`, 'warning');
            // TODO: Implement position closing API call
        }
    }

    // Portfolio utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updatePnLElement(id, value, isPercentage = false) {
        const element = document.getElementById(id);
        if (!element) return;

        element.textContent = isPercentage ? 
            (typeof value === 'number' ? (value * 100).toFixed(2) + '%' : value) :
            (typeof value === 'number' ? this.formatCurrency(value) : value);
        
        // Add color class
        element.className = element.className.replace(/\b(positive|negative|neutral)\b/g, '');
        const numericValue = typeof value === 'number' ? value : parseFloat(value);
        element.classList.add(this.getPnLClass(numericValue));
    }

    getPnLClass(value) {
        const numericValue = typeof value === 'number' ? value : parseFloat(value);
        if (numericValue > 0) return 'positive';
        if (numericValue < 0) return 'negative';
        return 'neutral';
    }

    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2
        }).format(value);
    }

    formatQuantity(value) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 4
        }).format(Math.abs(value));
    }

    formatAssetType(type) {
        const typeMap = {
            'equity': 'Stock',
            'futures': 'Future',
            'crypto': 'Crypto',
            'option': 'Option'
        };
        return typeMap[type] || type.charAt(0).toUpperCase() + type.slice(1);
    }

    parseMoneyValue(text) {
        return parseFloat(text.replace(/[$,]/g, ''));
    }

    showPortfolioError(section, message) {
        console.error(`Portfolio error in ${section}:`, message);
        this.showMessage(message, 'error');
    }

    showPositionsError(message) {
        const loadingEl = document.getElementById('positions-loading');
        if (loadingEl) {
            loadingEl.innerHTML = `<div class="error-message">${message}</div>`;
        }
    }

    showPerformanceError(message) {
        const loadingEl = document.getElementById('performance-loading');
        if (loadingEl) {
            loadingEl.innerHTML = `<div class="error-message">${message}</div>`;
        }
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