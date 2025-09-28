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
        
        // Module configuration
        this.moduleConfig = {};
        this.localModuleOverrides = this.loadLocalModuleOverrides();
        
        this.init();
    }

    async init() {
        console.log('🚀 AlgoTrendy Dashboard initializing...');
        
        // Load module configuration first
        await this.loadModuleConfiguration();
        
        // Set up navigation with module awareness
        this.setupNavigation();
        
        // Check backend connection
        await this.checkBackendStatus();
        
        // Start periodic status checks
        this.startStatusMonitoring();
        
        // Setup action buttons
        this.setupActionButtons();
        
        // Load trading system data when sections are activated
        this.setupTradingDataLoaders();
        
        // Setup module settings interface
        this.setupModuleSettings();
        
        // Initialize charts
        this.initializeCharts();
        
        console.log('✅ Dashboard initialized successfully');
    }
    
    async loadModuleConfiguration() {
        try {
            console.log('📦 Loading module configuration...');
            const response = await fetch('/api/config/modules');
            
            if (response.ok) {
                const config = await response.json();
                this.moduleConfig = config.modules || {};
                
                // Apply local overrides from localStorage
                this.applyLocalModuleOverrides();
                
                // Log module status
                this.logModuleStatus();
                
                // Update UI based on module status
                this.updateUIForModules();
            } else {
                console.warn('⚠️ Failed to load module config, using defaults');
                this.useDefaultModuleConfig();
            }
        } catch (error) {
            console.error('❌ Error loading module configuration:', error);
            this.useDefaultModuleConfig();
        }
    }
    
    useDefaultModuleConfig() {
        // Default all modules to enabled if backend is unavailable
        this.moduleConfig = {
            portfolio: { id: 'portfolio', name: 'Portfolio', enabled: true },
            trading: { id: 'trading', name: 'Trading', enabled: true },
            market_data: { id: 'market_data', name: 'Market Data', enabled: true },
            strategies: { id: 'strategies', name: 'Strategies', enabled: true },
            backtesting: { id: 'backtesting', name: 'Backtesting', enabled: true },
            ml_models: { id: 'ml_models', name: 'ML Models', enabled: true },
            stocks: { id: 'stocks', name: 'Stocks', enabled: true },
            futures: { id: 'futures', name: 'Futures', enabled: true },
            crypto: { id: 'crypto', name: 'Crypto', enabled: true },
            signals: { id: 'signals', name: 'Trading Signals', enabled: true }
        };
    }
    
    loadLocalModuleOverrides() {
        const saved = localStorage.getItem('moduleOverrides');
        return saved ? JSON.parse(saved) : {};
    }
    
    saveLocalModuleOverrides() {
        localStorage.setItem('moduleOverrides', JSON.stringify(this.localModuleOverrides));
    }
    
    applyLocalModuleOverrides() {
        // Apply local overrides to module config
        for (const [moduleId, enabled] of Object.entries(this.localModuleOverrides)) {
            if (this.moduleConfig[moduleId]) {
                this.moduleConfig[moduleId].enabled = enabled;
                console.log(`🔧 Applied local override: ${moduleId} = ${enabled ? 'enabled' : 'disabled'}`);
            }
        }
    }
    
    logModuleStatus() {
        console.log('📊 MODULE STATUS:');
        console.log('================');
        
        for (const [moduleId, config] of Object.entries(this.moduleConfig)) {
            const status = config.enabled ? '✅ ENABLED' : '❌ DISABLED';
            console.log(`  ${moduleId}: ${status} - ${config.description || ''}`);
        }
        
        console.log('================');
        
        const enabledCount = Object.values(this.moduleConfig).filter(m => m.enabled).length;
        const totalCount = Object.keys(this.moduleConfig).length;
        console.log(`📈 ${enabledCount}/${totalCount} modules enabled`);
    }
    
    isModuleEnabled(moduleId) {
        const module = this.moduleConfig[moduleId];
        return module && module.enabled && (module.dependenciesMet !== false);
    }
    
    updateUIForModules() {
        // Update navigation links based on module status
        const navLinks = document.querySelectorAll('.nav-link[data-module]');
        
        navLinks.forEach(link => {
            const moduleId = link.getAttribute('data-module');
            if (moduleId && !this.isModuleEnabled(moduleId)) {
                link.classList.add('module-disabled');
                link.setAttribute('title', `${this.moduleConfig[moduleId]?.name || moduleId} module is disabled`);
            }
        });
        
        // Update module status indicators
        this.updateModuleStatusIndicators();
    }
    
    updateModuleStatusIndicators() {
        // Update any module status badges or indicators in the UI
        const indicators = document.querySelectorAll('.module-status-indicator');
        
        indicators.forEach(indicator => {
            const moduleId = indicator.getAttribute('data-module');
            if (moduleId) {
                const isEnabled = this.isModuleEnabled(moduleId);
                indicator.classList.toggle('enabled', isEnabled);
                indicator.classList.toggle('disabled', !isEnabled);
                indicator.textContent = isEnabled ? 'Active' : 'Disabled';
            }
        });
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const contentSections = document.querySelectorAll('.content-section');

        navLinks.forEach(link => {
            link.addEventListener('click', async (e) => {
                e.preventDefault();
                
                const targetId = link.getAttribute('href').substring(1);
                const moduleId = link.getAttribute('data-module');
                
                // Check if module is enabled
                if (moduleId && !this.isModuleEnabled(moduleId)) {
                    this.showModuleDisabledMessage(moduleId);
                    return;
                }
                
                // Update active nav link
                navLinks.forEach(l => {
                    l.classList.remove('active');
                    l.removeAttribute('aria-current');
                });
                link.classList.add('active');
                // Mark the active link for assistive tech
                link.setAttribute('aria-current', 'page');
                
                // Show target content section
                contentSections.forEach(section => {
                    section.classList.remove('active');
                });
                
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add('active');
                        // Move focus to the region for keyboard users
                        targetSection.setAttribute('tabindex', '-1');
                        targetSection.focus({ preventScroll: true });
                    
                    // Handle settings section specially
                    if (targetId === 'settings') {
                        this.loadSettingsPanel();
                        return;
                    }
                    
                    // Load data for AI Systems sections and Portfolio
                    switch(targetId) {
                        case 'ml-models':
                            if (this.isModuleEnabled('ml_models')) {
                                await this.loadMLModels();
                            }
                            break;
                        case 'strategies':
                            if (this.isModuleEnabled('strategies')) {
                                await this.loadStrategies();
                            }
                            break;
                        case 'backtesting':
                            if (this.isModuleEnabled('backtesting')) {
                                await this.loadBacktests();
                            }
                            break;
                        case 'portfolio':
                            if (this.isModuleEnabled('portfolio')) {
                                await this.loadPortfolioData();
                            }
                            break;
                    }
                }
            });
        });
    }
    
    showModuleDisabledMessage(moduleId) {
        const module = this.moduleConfig[moduleId];
        const moduleName = module?.name || moduleId;
        
        const message = `
            <div class="module-disabled-message">
                <i class="fas fa-lock"></i>
                <h3>${moduleName} Module Disabled</h3>
                <p>${module?.description || 'This module is currently disabled.'}</p>
                <p>To enable this module, go to <a href="#settings" class="settings-link">Settings</a> and toggle the module on.</p>
            </div>
        `;
        
        this.showMessage(message, 'warning', 5000);
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
                        if (this.isModuleEnabled('trading')) {
                            this.showMessage('Trading interface would launch here...', 'info');
                        } else {
                            this.showModuleDisabledMessage('trading');
                        }
                        break;
                    case 'View Backtests':
                        if (this.isModuleEnabled('backtesting')) {
                            this.showMessage('Backtest results would display here...', 'info');
                        } else {
                            this.showModuleDisabledMessage('backtesting');
                        }
                        break;
                    case 'System Settings':
                        // Navigate to settings section
                        const settingsLink = document.querySelector('.nav-link[href="#settings"]');
                        if (settingsLink) {
                            settingsLink.click();
                        } else {
                            this.showMessage('Settings panel would open here...', 'info');
                        }
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

        // Accessibility: mark region as busy during load
        if (loadingEl) loadingEl.setAttribute('aria-busy', 'true');
        if (contentEl) contentEl.setAttribute('aria-hidden', 'true');
        
        try {
            const response = await fetch('/api/trading/models');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.renderMLModels(data.models);
            this.dataLoaded.models = true;
            
            if (loadingEl) {
                loadingEl.style.display = 'none';
                loadingEl.removeAttribute('aria-busy');
            }
            if (contentEl) {
                contentEl.style.display = 'grid';
                contentEl.removeAttribute('aria-hidden');
            }
            
        } catch (error) {
            console.error('Failed to load ML models:', error);
            if (loadingEl) loadingEl.innerHTML = `<div class="error-message">Failed to load ML models: ${error.message}</div>`;
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

            const backtests = Array.isArray(data?.backtests) ? data.backtests : [];
            this.renderBacktests(backtests);
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
        if (!container) {
            return;
        }

        const items = Array.isArray(backtests) ? backtests : [];
        container.innerHTML = '';

        if (items.length === 0) {
            container.innerHTML = '<div class="text-sm text-slate-500">No backtests available.</div>';
            return;
        }

        items.forEach(backtest => {
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
            
            if (loadingEl) {
                loadingEl.style.display = 'block';
                loadingEl.setAttribute('aria-busy', 'true');
            }
            if (contentEl) {
                contentEl.style.display = 'none';
                contentEl.setAttribute('aria-hidden', 'true');
            }

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
            if (loadingEl) loadingEl.removeAttribute('aria-busy');
            if (contentEl) contentEl.removeAttribute('aria-hidden');
            
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
                        <button class="action-btn small" type="button" title="View ${position.symbol} details" aria-label="View ${position.symbol} details" onclick="dashboard.viewPositionDetails('${position.symbol}')">
                            <i class="fas fa-eye" aria-hidden="true"></i>
                        </button>
                        <button class="action-btn small warning" type="button" title="Close position in ${position.symbol}" aria-label="Close position in ${position.symbol}" onclick="dashboard.closePosition('${position.symbol}')">
                            <i class="fas fa-times" aria-hidden="true"></i>
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
        
        // Create a new interval
        this.portfolioRefreshInterval = setInterval(() => {
            // Only refresh if portfolio tab is active
            const portfolioSection = document.getElementById('portfolio');
            if (portfolioSection && portfolioSection.classList.contains('active')) {
                this.loadPortfolioOverview();
                this.loadPositions();
            }
        }, 30000);

        // Ensure we only add the visibility handler once
        if (!this.visibilityHandlerAdded) {
            this.visibilityHandlerAdded = true;
            this.visibilityHandler = () => {
                if (document.hidden) {
                    // Pause polling while hidden
                    if (this.portfolioRefreshInterval) {
                        clearInterval(this.portfolioRefreshInterval);
                        this.portfolioRefreshInterval = null;
                    }
                } else {
                    // Resume and refresh immediately when visible again
                    this.loadPortfolioOverview();
                    this.loadPositions();
                    // restart interval
                    this.startPortfolioAutoRefresh();
                }
            };

            document.addEventListener('visibilitychange', this.visibilityHandler);
        }
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

// Module Settings extension for TradingDashboard class
Object.assign(TradingDashboard.prototype, {
    setupModuleSettings() {
        console.log('📋 Module settings interface setup');
    },
    
    loadSettingsPanel() {
        const settingsContent = document.getElementById('settings-content');
        if (!settingsContent) {
            return;
        }

        if (settingsContent.hasAttribute('hx-get')) {
            // HTMX will populate this section, so no JS rendering needed.
            return;
        }

        settingsContent.innerHTML = '<div class="text-sm text-slate-500">Module settings unavailable.</div>';
    },
    
    setupModuleToggleListeners() {
        const toggles = document.querySelectorAll('.module-toggle');
        toggles.forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const moduleId = e.target.getAttribute('data-module');
                const enabled = e.target.checked;
                this.localModuleOverrides[moduleId] = enabled;
                
                // Update toggle visual
                const slider = toggle.nextElementSibling;
                const sliderButton = slider.querySelector('span');
                slider.style.backgroundColor = enabled ? '#4CAF50' : '#ccc';
                sliderButton.style.left = enabled ? '30px' : '4px';
                
                console.log(`🔄 Module ${moduleId} set to ${enabled ? 'enabled' : 'disabled'}`);
            });
        });
        
        const applyBtn = document.getElementById('apply-module-settings');
        if (applyBtn) {
            applyBtn.addEventListener('click', async () => {
                await this.applyModuleSettings();
            });
        }
        
        const resetBtn = document.getElementById('reset-module-settings');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetModuleSettings();
            });
        }
    },
    
    async applyModuleSettings() {
        this.saveLocalModuleOverrides();
        
        for (const [moduleId, enabled] of Object.entries(this.localModuleOverrides)) {
            if (this.moduleConfig[moduleId] && this.moduleConfig[moduleId].enabled !== enabled) {
                try {
                    await fetch(`/api/config/modules/${moduleId}/toggle`, { method: 'POST' });
                } catch (error) {
                    console.warn(`Failed to toggle module ${moduleId}:`, error);
                }
            }
        }
        
        await this.loadModuleConfiguration();
        this.showMessage('Module settings applied successfully! Page will refresh...', 'success');
        
        // Refresh page after 2 seconds to apply changes
        setTimeout(() => {
            window.location.reload();
        }, 2000);
    },
    
    resetModuleSettings() {
        this.localModuleOverrides = {};
        localStorage.removeItem('moduleOverrides');
        
        this.loadModuleConfiguration().then(() => {
            this.loadSettingsPanel();
            this.showMessage('Module settings reset to defaults', 'info');
        });
    },

    // =============================================================================
    // COMPREHENSIVE CHARTING SYSTEM
    // =============================================================================

    initializeCharts() {
        console.log('📊 Initializing charts...');
        
        this.charts = {};
        this.chartConfig = this.getChartConfig();
        
        // Initialize charts when sections become active
        this.setupChartInitializationObserver();
        
        // Initialize overview charts immediately since overview is active by default
        setTimeout(() => {
            this.initializeOverviewCharts();
        }, 500);
        
        console.log('📈 Chart system initialized');
    },

    getChartConfig() {
        return {
            colors: {
                primary: '#2563eb',
                success: '#059669',
                danger: '#dc2626',
                warning: '#d97706',
                secondary: '#64748b',
                muted: '#94a3b8',
                chartBlue: '#3b82f6',
                chartGreen: '#10b981',
                chartRed: '#ef4444',
                chartYellow: '#f59e0b',
                chartPurple: '#8b5cf6',
                chartGray: '#6b7280'
            },
            fonts: {
                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
                size: 11,
                weight: 500,
                color: '#475569'
            },
            grid: {
                color: '#e2e8f0',
                borderWidth: 1
            },
            defaultOptions: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            padding: 15,
                            font: {
                                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
                                size: 10,
                                weight: 500
                            },
                            color: '#475569',
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: '#ffffff',
                        titleColor: '#0f172a',
                        bodyColor: '#475569',
                        borderColor: '#cbd5e1',
                        borderWidth: 2,
                        cornerRadius: 0,
                        padding: 12,
                        displayColors: false,
                        titleFont: {
                            size: 11,
                            weight: 600
                        },
                        bodyFont: {
                            size: 10,
                            weight: 500,
                            family: 'SF Mono, Monaco, Inconsolata, Courier New, monospace'
                        }
                    }
                },
                scales: {
                    x: {
                        border: {
                            color: '#cbd5e1',
                            width: 2
                        },
                        grid: {
                            color: '#e2e8f0',
                            drawBorder: true
                        },
                        ticks: {
                            color: '#475569',
                            font: {
                                size: 10,
                                weight: 500,
                                family: 'SF Mono, Monaco, Inconsolata, Courier New, monospace'
                            },
                            padding: 8
                        }
                    },
                    y: {
                        border: {
                            color: '#cbd5e1',
                            width: 2
                        },
                        grid: {
                            color: '#e2e8f0',
                            drawBorder: true
                        },
                        ticks: {
                            color: '#475569',
                            font: {
                                size: 10,
                                weight: 500,
                                family: 'SF Mono, Monaco, Inconsolata, Courier New, monospace'
                            },
                            padding: 8
                        }
                    }
                }
            }
        };
    },

    setupChartInitializationObserver() {
        // Observer to initialize charts when sections become visible
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    this.initializeSectionCharts(sectionId);
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.content-section').forEach(section => {
            observer.observe(section);
        });

        // Also set up navigation listeners to initialize charts when sections are clicked
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                const targetSection = e.target.getAttribute('href')?.substring(1);
                if (targetSection) {
                    setTimeout(() => {
                        this.initializeSectionCharts(targetSection);
                    }, 100);
                }
            });
        });
    },

    initializeSectionCharts(sectionId) {
        console.log(`📊 Initializing charts for section: ${sectionId}`);
        
        switch (sectionId) {
            case 'overview':
                this.initializeOverviewCharts();
                break;
            case 'portfolio':
                this.initializePortfolioCharts();
                break;
            case 'ml-models':
                this.initializeMLCharts();
                break;
            case 'strategies':
                this.initializeStrategyCharts();
                break;
            case 'backtesting':
                this.initializeBacktestCharts();
                break;
        }
    },

    // =============================================================================
    // OVERVIEW CHARTS
    // =============================================================================

    initializeOverviewCharts() {
        this.createOverviewPerformanceChart();
        this.createOverviewAllocationChart();
        this.updateOverviewStats();
    },

    createOverviewPerformanceChart() {
        const ctx = document.getElementById('overview-performance-chart');
        if (!ctx || this.charts['overview-performance']) return;

        const data = this.generatePerformanceData(30);
        
        this.charts['overview-performance'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Portfolio Value',
                    data: data.values,
                    borderColor: this.chartConfig.colors.primary,
                    backgroundColor: this.chartConfig.colors.primary + '10',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '$' + (value / 1000).toFixed(0) + 'K';
                            }
                        }
                    }
                }
            }
        });
    },

    createOverviewAllocationChart() {
        const ctx = document.getElementById('overview-allocation-chart');
        if (!ctx || this.charts['overview-allocation']) return;

        const data = this.generateAllocationData();
        
        this.charts['overview-allocation'] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        this.chartConfig.colors.chartBlue,
                        this.chartConfig.colors.chartGreen,
                        this.chartConfig.colors.chartYellow,
                        this.chartConfig.colors.chartPurple,
                        this.chartConfig.colors.chartGray
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                cutout: '60%',
                plugins: {
                    ...this.chartConfig.defaultOptions.plugins,
                    legend: {
                        ...this.chartConfig.defaultOptions.plugins.legend,
                        position: 'right'
                    },
                    tooltip: {
                        ...this.chartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    // =============================================================================
    // PORTFOLIO CHARTS
    // =============================================================================

    initializePortfolioCharts() {
        this.createPortfolioPerformanceChart();
        this.createPortfolioAllocationChart();
        this.createPortfolioPnLChart();
        this.createPortfolioRiskChart();
    },

    createPortfolioPerformanceChart() {
        const ctx = document.getElementById('portfolio-performance-chart');
        if (!ctx || this.charts['portfolio-performance']) return;

        const data = this.generatePerformanceData(90);
        
        this.charts['portfolio-performance'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Portfolio Performance',
                    data: data.values,
                    borderColor: this.chartConfig.colors.primary,
                    backgroundColor: this.chartConfig.colors.primary + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true
                }, {
                    label: 'Benchmark (S&P 500)',
                    data: data.benchmark,
                    borderColor: this.chartConfig.colors.secondary,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5]
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '$' + (value / 1000).toFixed(0) + 'K';
                            }
                        }
                    }
                }
            }
        });
    },

    createPortfolioAllocationChart() {
        const ctx = document.getElementById('portfolio-allocation-chart');
        if (!ctx || this.charts['portfolio-allocation']) return;

        const data = this.generateDetailedAllocationData();
        
        this.charts['portfolio-allocation'] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: data.colors,
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                cutout: '65%',
                plugins: {
                    ...this.chartConfig.defaultOptions.plugins,
                    legend: {
                        ...this.chartConfig.defaultOptions.plugins.legend,
                        position: 'right'
                    }
                }
            }
        });
    },

    createPortfolioPnLChart() {
        const ctx = document.getElementById('portfolio-pnl-chart');
        if (!ctx || this.charts['portfolio-pnl']) return;

        const data = this.generatePnLData();
        
        this.charts['portfolio-pnl'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Daily P&L',
                    data: data.values,
                    backgroundColor: data.values.map(v => 
                        v >= 0 ? this.chartConfig.colors.success + '80' : this.chartConfig.colors.danger + '80'
                    ),
                    borderColor: data.values.map(v => 
                        v >= 0 ? this.chartConfig.colors.success : this.chartConfig.colors.danger
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    },

    createPortfolioRiskChart() {
        const ctx = document.getElementById('portfolio-risk-chart');
        if (!ctx || this.charts['portfolio-risk']) return;

        const data = this.generateRiskData();
        
        this.charts['portfolio-risk'] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Volatility', 'Correlation', 'Beta', 'Sharpe Ratio', 'Max Drawdown', 'VaR'],
                datasets: [{
                    label: 'Current Portfolio',
                    data: data.current,
                    borderColor: this.chartConfig.colors.primary,
                    backgroundColor: this.chartConfig.colors.primary + '20',
                    borderWidth: 2,
                    pointBackgroundColor: this.chartConfig.colors.primary,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }, {
                    label: 'Target Risk Profile',
                    data: data.target,
                    borderColor: this.chartConfig.colors.secondary,
                    backgroundColor: this.chartConfig.colors.secondary + '10',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointBackgroundColor: this.chartConfig.colors.secondary,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: this.chartConfig.grid.color
                        },
                        angleLines: {
                            color: this.chartConfig.grid.color
                        },
                        pointLabels: {
                            font: {
                                size: 10,
                                weight: 500
                            },
                            color: this.chartConfig.fonts.color
                        },
                        ticks: {
                            display: false
                        }
                    }
                }
            }
        });
    },

    // =============================================================================
    // ML MODEL CHARTS
    // =============================================================================

    initializeMLCharts() {
        this.createMLPerformanceChart();
        this.createMLTrainingChart();
        this.createMLFeatureImportanceChart();
    },

    createMLPerformanceChart() {
        const ctx = document.getElementById('ml-performance-chart');
        if (!ctx || this.charts['ml-performance']) return;

        const data = this.generateMLPerformanceData();
        
        this.charts['ml-performance'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Accuracy',
                    data: data.accuracy,
                    backgroundColor: this.chartConfig.colors.chartBlue + '80',
                    borderColor: this.chartConfig.colors.chartBlue,
                    borderWidth: 2
                }, {
                    label: 'Precision',
                    data: data.precision,
                    backgroundColor: this.chartConfig.colors.chartGreen + '80',
                    borderColor: this.chartConfig.colors.chartGreen,
                    borderWidth: 2
                }, {
                    label: 'Recall',
                    data: data.recall,
                    backgroundColor: this.chartConfig.colors.chartYellow + '80',
                    borderColor: this.chartConfig.colors.chartYellow,
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    createMLTrainingChart() {
        const ctx = document.getElementById('ml-training-chart');
        if (!ctx || this.charts['ml-training']) return;

        const data = this.generateTrainingData();
        
        this.charts['ml-training'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [{
                    label: 'Training Loss',
                    data: data.trainLoss,
                    borderColor: this.chartConfig.colors.danger,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 2,
                    yAxisID: 'y'
                }, {
                    label: 'Validation Loss',
                    data: data.valLoss,
                    borderColor: this.chartConfig.colors.warning,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 2,
                    borderDash: [3, 3],
                    yAxisID: 'y'
                }, {
                    label: 'Accuracy',
                    data: data.accuracy,
                    borderColor: this.chartConfig.colors.success,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 2,
                    yAxisID: 'y1'
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss',
                            font: { size: 10, weight: 600 },
                            color: this.chartConfig.fonts.color
                        }
                    },
                    y1: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            font: { size: 10, weight: 600 },
                            color: this.chartConfig.fonts.color
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    createMLFeatureImportanceChart() {
        const ctx = document.getElementById('ml-feature-importance-chart');
        if (!ctx || this.charts['ml-feature-importance']) return;

        const data = this.generateFeatureImportanceData();
        
        this.charts['ml-feature-importance'] = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: data.features,
                datasets: [{
                    label: 'Feature Importance',
                    data: data.importance,
                    backgroundColor: this.chartConfig.colors.primary + '80',
                    borderColor: this.chartConfig.colors.primary,
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                indexAxis: 'y',
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    x: {
                        ...this.chartConfig.defaultOptions.scales.x,
                        beginAtZero: true,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.x.ticks,
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    // =============================================================================
    // STRATEGY & BACKTESTING CHARTS
    // =============================================================================

    initializeStrategyCharts() {
        this.createStrategyPerformanceChart();
        this.createStrategyWinLossChart();
        this.createStrategyRiskReturnChart();
    },

    createStrategyPerformanceChart() {
        const ctx = document.getElementById('strategy-performance-chart');
        if (!ctx || this.charts['strategy-performance']) return;

        const data = this.generateStrategyPerformanceData();
        
        this.charts['strategy-performance'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.strategies.map((strategy, index) => ({
                    label: strategy.name,
                    data: strategy.performance,
                    borderColor: [
                        this.chartConfig.colors.primary,
                        this.chartConfig.colors.success,
                        this.chartConfig.colors.warning,
                        this.chartConfig.colors.chartPurple
                    ][index % 4],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }))
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    createStrategyWinLossChart() {
        const ctx = document.getElementById('strategy-winloss-chart');
        if (!ctx || this.charts['strategy-winloss']) return;

        const data = this.generateWinLossData();
        
        this.charts['strategy-winloss'] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Winning Trades', 'Losing Trades'],
                datasets: [{
                    data: [data.wins, data.losses],
                    backgroundColor: [
                        this.chartConfig.colors.success,
                        this.chartConfig.colors.danger
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                cutout: '60%',
                plugins: {
                    ...this.chartConfig.defaultOptions.plugins,
                    legend: {
                        ...this.chartConfig.defaultOptions.plugins.legend,
                        position: 'bottom'
                    },
                    tooltip: {
                        ...this.chartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const percentage = ((context.parsed / (data.wins + data.losses)) * 100).toFixed(1);
                                return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                            }
                        }
                    }
                }
            }
        });
    },

    createStrategyRiskReturnChart() {
        const ctx = document.getElementById('strategy-risk-return-chart');
        if (!ctx || this.charts['strategy-risk-return']) return;

        const data = this.generateRiskReturnData();
        
        this.charts['strategy-risk-return'] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Strategies',
                    data: data.points,
                    backgroundColor: this.chartConfig.colors.primary + '80',
                    borderColor: this.chartConfig.colors.primary,
                    borderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    x: {
                        ...this.chartConfig.defaultOptions.scales.x,
                        title: {
                            display: true,
                            text: 'Risk (Volatility %)',
                            font: { size: 11, weight: 600 },
                            color: this.chartConfig.fonts.color
                        },
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.x.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    },
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Return (%)',
                            font: { size: 11, weight: 600 },
                            color: this.chartConfig.fonts.color
                        },
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    ...this.chartConfig.defaultOptions.plugins,
                    tooltip: {
                        ...this.chartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return `Risk: ${context.parsed.x.toFixed(1)}%, Return: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    },

    initializeBacktestCharts() {
        this.createBacktestReturnsChart();
        this.createBacktestDrawdownChart();
        this.createBacktestDistributionChart();
    },

    createBacktestReturnsChart() {
        const ctx = document.getElementById('backtest-returns-chart');
        if (!ctx || this.charts['backtest-returns']) return;

        const data = this.generateBacktestReturnsData();
        
        this.charts['backtest-returns'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Cumulative Returns',
                    data: data.returns,
                    borderColor: this.chartConfig.colors.primary,
                    backgroundColor: this.chartConfig.colors.primary + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    createBacktestDrawdownChart() {
        const ctx = document.getElementById('backtest-drawdown-chart');
        if (!ctx || this.charts['backtest-drawdown']) return;

        const data = this.generateDrawdownData();
        
        this.charts['backtest-drawdown'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Drawdown',
                    data: data.drawdown,
                    borderColor: this.chartConfig.colors.danger,
                    backgroundColor: this.chartConfig.colors.danger + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        max: 0,
                        ticks: {
                            ...this.chartConfig.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    createBacktestDistributionChart() {
        const ctx = document.getElementById('backtest-distribution-chart');
        if (!ctx || this.charts['backtest-distribution']) return;

        const data = this.generateDistributionData();
        
        this.charts['backtest-distribution'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.bins,
                datasets: [{
                    label: 'Frequency',
                    data: data.frequency,
                    backgroundColor: this.chartConfig.colors.primary + '80',
                    borderColor: this.chartConfig.colors.primary,
                    borderWidth: 2
                }]
            },
            options: {
                ...this.chartConfig.defaultOptions,
                scales: {
                    ...this.chartConfig.defaultOptions.scales,
                    x: {
                        ...this.chartConfig.defaultOptions.scales.x,
                        title: {
                            display: true,
                            text: 'Monthly Returns (%)',
                            font: { size: 11, weight: 600 },
                            color: this.chartConfig.fonts.color
                        }
                    },
                    y: {
                        ...this.chartConfig.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Frequency',
                            font: { size: 11, weight: 600 },
                            color: this.chartConfig.fonts.color
                        }
                    }
                }
            }
        });
    },

    // =============================================================================
    // DATA GENERATION METHODS
    // =============================================================================

    generatePerformanceData(days) {
        const labels = [];
        const values = [];
        const benchmark = [];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        
        let currentValue = 100000;
        let benchmarkValue = 100000;
        
        for (let i = 0; i <= days; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            labels.push(date.toLocaleDateString());
            
            // Generate realistic performance data with some volatility
            const change = (Math.random() - 0.48) * 0.02; // Slight upward bias
            currentValue *= (1 + change);
            values.push(Math.round(currentValue));
            
            const benchmarkChange = (Math.random() - 0.49) * 0.015;
            benchmarkValue *= (1 + benchmarkChange);
            benchmark.push(Math.round(benchmarkValue));
        }
        
        return { labels, values, benchmark };
    },

    generateAllocationData() {
        return {
            labels: ['Stocks', 'Bonds', 'Commodities', 'Crypto', 'Cash'],
            values: [45.2, 25.8, 12.3, 8.9, 7.8]
        };
    },

    generateDetailedAllocationData() {
        return {
            labels: ['Technology', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Goods', 'Bonds', 'Commodities', 'Cash'],
            values: [18.5, 12.7, 14.0, 8.2, 11.8, 25.8, 6.3, 2.7],
            colors: [
                this.chartConfig.colors.chartBlue,
                this.chartConfig.colors.chartGreen,
                this.chartConfig.colors.primary,
                this.chartConfig.colors.warning,
                this.chartConfig.colors.chartPurple,
                this.chartConfig.colors.secondary,
                this.chartConfig.colors.chartYellow,
                this.chartConfig.colors.chartGray
            ]
        };
    },

    generatePnLData() {
        const labels = [];
        const values = [];
        
        for (let i = 0; i < 30; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (29 - i));
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            // Generate P&L data with some winning and losing days
            const pnl = (Math.random() - 0.45) * 5000; // Slight positive bias
            values.push(Math.round(pnl));
        }
        
        return { labels, values };
    },

    generateRiskData() {
        return {
            current: [75, 45, 88, 92, 65, 78],
            target: [60, 40, 70, 85, 50, 60]
        };
    },

    generateMLPerformanceData() {
        return {
            labels: ['XGBoost', 'Random Forest', 'LSTM', 'SVM', 'Linear Regression'],
            accuracy: [87.3, 84.1, 91.2, 79.8, 76.5],
            precision: [85.7, 82.9, 89.4, 77.2, 74.8],
            recall: [89.1, 85.6, 93.1, 82.3, 78.9]
        };
    },

    generateTrainingData() {
        const epochs = [];
        const trainLoss = [];
        const valLoss = [];
        const accuracy = [];
        
        for (let i = 1; i <= 50; i++) {
            epochs.push(i);
            // Decreasing loss with some noise
            trainLoss.push(Math.max(0.05, 2.5 * Math.exp(-i/15) + Math.random() * 0.1));
            valLoss.push(Math.max(0.08, 2.7 * Math.exp(-i/18) + Math.random() * 0.15));
            // Increasing accuracy
            accuracy.push(Math.min(98, 60 + 35 * (1 - Math.exp(-i/12)) + (Math.random() - 0.5) * 2));
        }
        
        return { epochs, trainLoss, valLoss, accuracy };
    },

    generateFeatureImportanceData() {
        return {
            features: ['Price Momentum', 'Volume', 'RSI', 'MACD', 'Moving Average', 'Volatility', 'Market Cap', 'P/E Ratio'],
            importance: [0.28, 0.22, 0.15, 0.12, 0.08, 0.07, 0.05, 0.03]
        };
    },

    generateStrategyPerformanceData() {
        const labels = [];
        const strategies = [
            { name: 'Momentum Strategy', performance: [] },
            { name: 'Mean Reversion', performance: [] },
            { name: 'Pairs Trading', performance: [] },
            { name: 'ML Enhanced', performance: [] }
        ];
        
        for (let i = 0; i < 12; i++) {
            const date = new Date();
            date.setMonth(date.getMonth() - (11 - i));
            labels.push(date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
            
            // Generate cumulative performance for each strategy
            strategies[0].performance.push(5 + i * 2.5 + Math.random() * 3); // Momentum
            strategies[1].performance.push(3 + i * 1.8 + Math.random() * 2.5); // Mean reversion
            strategies[2].performance.push(4 + i * 2.2 + Math.random() * 3.5); // Pairs trading
            strategies[3].performance.push(6 + i * 3.1 + Math.random() * 4); // ML Enhanced
        }
        
        return { labels, strategies };
    },

    generateWinLossData() {
        return {
            wins: 127,
            losses: 48
        };
    },

    generateRiskReturnData() {
        const strategies = ['Momentum', 'Mean Reversion', 'Arbitrage', 'ML Enhanced', 'Pairs Trading', 'Trend Following'];
        const points = strategies.map(() => ({
            x: Math.random() * 15 + 5, // Risk (5-20%)
            y: Math.random() * 25 + 5   // Return (5-30%)
        }));
        
        return { points };
    },

    generateBacktestReturnsData() {
        const labels = [];
        const returns = [];
        let cumulative = 0;
        
        for (let i = 0; i < 252; i++) { // Trading days in a year
            const date = new Date();
            date.setDate(date.getDate() - (251 - i));
            if (i % 10 === 0) labels.push(date.toLocaleDateString('en-US', { month: 'short' }));
            
            const dailyReturn = (Math.random() - 0.47) * 2; // Slight positive bias
            cumulative += dailyReturn;
            if (i % 10 === 0) returns.push(cumulative);
        }
        
        return { labels, returns };
    },

    generateDrawdownData() {
        const labels = [];
        const drawdown = [];
        let peak = 0;
        let current = 0;
        
        for (let i = 0; i < 252; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (251 - i));
            if (i % 10 === 0) labels.push(date.toLocaleDateString('en-US', { month: 'short' }));
            
            const dailyReturn = (Math.random() - 0.47) * 2;
            current += dailyReturn;
            peak = Math.max(peak, current);
            
            if (i % 10 === 0) drawdown.push(current - peak);
        }
        
        return { labels, drawdown };
    },

    generateDistributionData() {
        const bins = ['-15%', '-10%', '-5%', '0%', '5%', '10%', '15%', '20%'];
        const frequency = [2, 5, 12, 18, 22, 15, 8, 3]; // Normal-like distribution
        
        return { bins, frequency };
    },

    updateOverviewStats() {
        // Update large statistical displays
        const elements = {
            'total-pnl': '$47,328',
            'win-rate': '73%',
            'active-positions': '12',
            'risk-score': '2.4',
            'ai-systems-count': '3 Models Active',
            'trading-systems-count': '12 Available',
            'risk-status': 'Active'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    },

    // Chart control methods
    setupChartControls() {
        document.querySelectorAll('.chart-control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const button = e.target;
                const container = button.closest('.chart-container');
                const chartCanvas = container.querySelector('canvas');
                
                // Update button states
                container.querySelectorAll('.chart-control-btn').forEach(b => b.classList.remove('active'));
                button.classList.add('active');
                
                // Update chart data based on selection
                this.updateChartData(chartCanvas.id, button.dataset);
            });
        });
    },

    updateChartData(chartId, dataset) {
        const chart = this.charts[chartId.replace('-chart', '')];
        if (!chart) return;
        
        // Implement chart updates based on control selections
        console.log(`Updating chart ${chartId} with:`, dataset);
        // This would be expanded to actually update chart data based on the controls
    }
});