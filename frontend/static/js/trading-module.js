/**
 * Trading Module for AlgoTrendy Dashboard
 * Complete trading functionality for stocks, futures, and crypto
 */

// Extend TradingDashboard with trading functionality
Object.assign(TradingDashboard.prototype, {
    // Initialize trading forms and event listeners
    initializeTradingForms() {
        console.log('💹 Initializing trading forms...');
        
        // Initialize for each trading type
        ['stocks', 'futures', 'crypto'].forEach(type => {
            this.initializeTradingForm(type);
            this.setupOrderTypeHandlers(type);
            this.setupSymbolAutocomplete(type);
            this.setupQuantityCalculators(type);
            this.setupOrderSideToggle(type);
            this.setupQuantityButtons(type);
        });
        
        // Initialize keyboard shortcuts
        this.setupKeyboardShortcuts();
    },
    
    // Setup trading event listeners
    setupTradingEventListeners() {
        console.log('📊 Setting up trading event listeners...');
        
        ['stocks', 'futures', 'crypto'].forEach(type => {
            // Form submission
            const form = document.getElementById(`${type}-order-form`);
            if (form) {
                form.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.submitOrder(type);
                });
            }
            
            // Watchlist management
            const addBtn = document.getElementById(`add-${type}-watchlist-btn`);
            if (addBtn) {
                addBtn.addEventListener('click', () => this.showAddToWatchlist(type));
            }
            
            // Order filters
            const statusFilter = document.getElementById(`${type}-status-filter`);
            if (statusFilter) {
                statusFilter.addEventListener('change', (e) => {
                    this.filterOrders(type, e.target.value);
                });
            }
            
            const dateFilter = document.getElementById(`${type}-date-filter`);
            if (dateFilter) {
                dateFilter.addEventListener('change', (e) => {
                    this.filterOrdersByDate(type, e.target.value);
                });
            }
            
            // Refresh button
            const refreshBtn = document.getElementById(`refresh-${type}-btn`);
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    this.loadTradingInterface(type);
                });
            }
        });
    },
    
    // Initialize individual trading form
    initializeTradingForm(type) {
        const symbolInput = document.getElementById(`${type}-symbol`);
        const quantityInput = document.getElementById(`${type}-quantity`);
        const priceInput = document.getElementById(`${type}-price`);
        
        // Symbol input change handler
        if (symbolInput) {
            symbolInput.addEventListener('input', debounce(() => {
                this.searchSymbol(type, symbolInput.value);
            }, 300));
            
            symbolInput.addEventListener('blur', () => {
                setTimeout(() => {
                    const dropdown = document.getElementById(`${type}-symbol-dropdown`);
                    if (dropdown) dropdown.style.display = 'none';
                }, 200);
            });
        }
        
        // Quantity input handler for cost calculation
        if (quantityInput) {
            quantityInput.addEventListener('input', () => {
                this.calculateOrderValue(type);
            });
        }
        
        // Price input handler for limit orders
        if (priceInput) {
            priceInput.addEventListener('input', () => {
                this.calculateOrderValue(type);
            });
        }
    },
    
    // Setup order type handlers
    setupOrderTypeHandlers(type) {
        const orderTypeSelect = document.getElementById(`${type}-order-type`);
        const priceGroup = document.getElementById(`${type}-price-group`);
        
        if (orderTypeSelect && priceGroup) {
            orderTypeSelect.addEventListener('change', (e) => {
                if (e.target.value === 'limit' || e.target.value === 'stop') {
                    priceGroup.style.display = 'block';
                } else {
                    priceGroup.style.display = 'none';
                }
                this.calculateOrderValue(type);
            });
        }
    },
    
    // Setup order side toggle (buy/sell)
    setupOrderSideToggle(type) {
        const sideButtons = document.querySelectorAll(`#${type} .side-btn`);
        const submitBtn = document.getElementById(`${type}-submit-btn`);
        
        sideButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Update active state
                sideButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update submit button text and style
                const side = btn.dataset.side;
                if (submitBtn) {
                    const btnText = submitBtn.querySelector('.btn-text');
                    if (btnText) {
                        btnText.textContent = side === 'buy' ? 'Place Buy Order' : 'Place Sell Order';
                    }
                    
                    // Update button color
                    if (side === 'buy') {
                        submitBtn.classList.remove('sell-order');
                        submitBtn.classList.add('buy-order');
                    } else {
                        submitBtn.classList.remove('buy-order');
                        submitBtn.classList.add('sell-order');
                    }
                }
                
                this.calculateOrderValue(type);
            });
        });
    },
    
    // Setup quantity quick buttons
    setupQuantityButtons(type) {
        const qtyButtons = document.querySelectorAll(`#${type} .qty-btn`);
        const quantityInput = document.getElementById(`${type}-quantity`);
        
        qtyButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const qty = parseFloat(btn.dataset.qty);
                if (quantityInput) {
                    quantityInput.value = qty;
                    this.calculateOrderValue(type);
                }
            });
        });
    },
    
    // Setup buying power percentage buttons
    setupQuantityCalculators(type) {
        // This would calculate based on buying power percentage
        // Implementation depends on backend API structure
    },
    
    // Setup symbol autocomplete
    setupSymbolAutocomplete(type) {
        const symbolInput = document.getElementById(`${type}-symbol`);
        const dropdown = document.getElementById(`${type}-symbol-dropdown`);
        
        if (!symbolInput || !dropdown) return;
        
        // Handle dropdown item clicks
        dropdown.addEventListener('click', (e) => {
            const item = e.target.closest('.symbol-item');
            if (item) {
                const symbol = item.dataset.symbol;
                symbolInput.value = symbol;
                dropdown.style.display = 'none';
                
                // Load symbol info
                this.loadSymbolInfo(type, symbol);
                
                // Calculate order value
                this.calculateOrderValue(type);
            }
        });
    },
    
    // Search symbols for autocomplete
    async searchSymbol(type, query) {
        if (!query || query.length < 1) {
            const dropdown = document.getElementById(`${type}-symbol-dropdown`);
            if (dropdown) dropdown.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/api/market/search?q=${encodeURIComponent(query)}&type=${type}`);
            const data = await response.json();
            
            if (data.symbols && data.symbols.length > 0) {
                this.showSymbolDropdown(type, data.symbols);
            } else {
                // For now, show mock suggestions
                this.showMockSymbolSuggestions(type, query);
            }
        } catch (error) {
            console.error('Symbol search error:', error);
            // Show mock suggestions as fallback
            this.showMockSymbolSuggestions(type, query);
        }
    },
    
    // Show mock symbol suggestions (fallback)
    showMockSymbolSuggestions(type, query) {
        const mockSymbols = {
            stocks: [
                { symbol: 'AAPL', name: 'Apple Inc.' },
                { symbol: 'GOOGL', name: 'Alphabet Inc.' },
                { symbol: 'MSFT', name: 'Microsoft Corporation' },
                { symbol: 'AMZN', name: 'Amazon.com Inc.' },
                { symbol: 'TSLA', name: 'Tesla Inc.' },
                { symbol: 'META', name: 'Meta Platforms Inc.' },
                { symbol: 'NVDA', name: 'NVIDIA Corporation' },
                { symbol: 'SPY', name: 'SPDR S&P 500 ETF' }
            ],
            futures: [
                { symbol: 'ES', name: 'E-mini S&P 500' },
                { symbol: 'NQ', name: 'E-mini Nasdaq-100' },
                { symbol: 'GC', name: 'Gold Futures' },
                { symbol: 'CL', name: 'Crude Oil Futures' },
                { symbol: 'YM', name: 'E-mini Dow' },
                { symbol: 'RTY', name: 'E-mini Russell 2000' }
            ],
            crypto: [
                { symbol: 'BTC-USD', name: 'Bitcoin' },
                { symbol: 'ETH-USD', name: 'Ethereum' },
                { symbol: 'ADA-USD', name: 'Cardano' },
                { symbol: 'SOL-USD', name: 'Solana' },
                { symbol: 'DOT-USD', name: 'Polkadot' },
                { symbol: 'MATIC-USD', name: 'Polygon' }
            ]
        };
        
        const filtered = mockSymbols[type].filter(s => 
            s.symbol.toLowerCase().includes(query.toLowerCase()) ||
            s.name.toLowerCase().includes(query.toLowerCase())
        );
        
        this.showSymbolDropdown(type, filtered);
    },
    
    // Show symbol dropdown
    showSymbolDropdown(type, symbols) {
        const dropdown = document.getElementById(`${type}-symbol-dropdown`);
        if (!dropdown) return;
        
        dropdown.innerHTML = '';
        
        symbols.forEach(symbol => {
            const item = document.createElement('div');
            item.className = 'symbol-item';
            item.dataset.symbol = symbol.symbol;
            item.innerHTML = `
                <span class="symbol-code">${symbol.symbol}</span>
                <span class="symbol-name">${symbol.name}</span>
            `;
            dropdown.appendChild(item);
        });
        
        dropdown.style.display = symbols.length > 0 ? 'block' : 'none';
    },
    
    // Load symbol info and current price
    async loadSymbolInfo(type, symbol) {
        const infoEl = document.getElementById(`${type}-symbol-info`);
        if (!infoEl) return;
        
        try {
            const response = await fetch(`/api/market/prices?symbols=${symbol}`);
            const data = await response.json();
            
            if (data.prices && data.prices[symbol]) {
                const price = data.prices[symbol];
                this.updateSymbolInfo(type, {
                    symbol: symbol,
                    price: price.last || 100,
                    change: price.change || 0,
                    changePercent: price.changePercent || 0,
                    bid: price.bid || (price.last - 0.01) || 99.99,
                    ask: price.ask || (price.last + 0.01) || 100.01
                });
                
                infoEl.style.display = 'block';
            } else {
                // Mock data fallback
                this.updateSymbolInfo(type, {
                    symbol: symbol,
                    price: 100 + Math.random() * 50,
                    change: (Math.random() - 0.5) * 10,
                    changePercent: (Math.random() - 0.5) * 5,
                    bid: 99.99,
                    ask: 100.01
                });
                
                infoEl.style.display = 'block';
            }
        } catch (error) {
            console.error('Error loading symbol info:', error);
            // Show mock data
            this.updateSymbolInfo(type, {
                symbol: symbol,
                price: 100,
                change: 1.5,
                changePercent: 1.52,
                bid: 99.99,
                ask: 100.01
            });
            
            infoEl.style.display = 'block';
        }
        
        // Store current price for calculations
        this.marketPrices[symbol] = infoEl.querySelector('.current-price')?.textContent.replace('$', '') || 100;
    },
    
    // Update symbol info display
    updateSymbolInfo(type, info) {
        const infoEl = document.getElementById(`${type}-symbol-info`);
        if (!infoEl) return;
        
        const priceEl = infoEl.querySelector('.current-price');
        const changeEl = infoEl.querySelector('.price-change');
        const bidEl = infoEl.querySelector('.bid-price');
        const askEl = infoEl.querySelector('.ask-price');
        
        if (priceEl) priceEl.textContent = `$${info.price.toFixed(2)}`;
        if (changeEl) {
            const sign = info.change >= 0 ? '+' : '';
            changeEl.textContent = `${sign}$${info.change.toFixed(2)} (${sign}${info.changePercent.toFixed(2)}%)`;
            changeEl.className = `price-change ${info.change >= 0 ? 'positive' : 'negative'}`;
        }
        if (bidEl) bidEl.textContent = `$${info.bid.toFixed(2)}`;
        if (askEl) askEl.textContent = `$${info.ask.toFixed(2)}`;
    },
    
    // Calculate order value
    calculateOrderValue(type) {
        const symbolInput = document.getElementById(`${type}-symbol`);
        const quantityInput = document.getElementById(`${type}-quantity`);
        const orderTypeSelect = document.getElementById(`${type}-order-type`);
        const priceInput = document.getElementById(`${type}-price`);
        const sideBtn = document.querySelector(`#${type} .side-btn.active`);
        
        const symbol = symbolInput?.value;
        const quantity = parseFloat(quantityInput?.value || 0);
        const orderType = orderTypeSelect?.value;
        const side = sideBtn?.dataset.side || 'buy';
        
        let price = 0;
        if (orderType === 'market') {
            // Use current market price
            price = parseFloat(this.marketPrices[symbol] || 100);
        } else if (orderType === 'limit' || orderType === 'stop') {
            // Use limit price
            price = parseFloat(priceInput?.value || 0);
        }
        
        const estimatedCost = price * quantity;
        const commission = type === 'stocks' ? 0 : (type === 'futures' ? 4.50 : 2.50);
        const total = estimatedCost + (side === 'buy' ? commission : -commission);
        
        // Update display
        const costEl = document.getElementById(`${type}-estimated-cost`);
        const commissionEl = document.getElementById(`${type}-commission`);
        const totalEl = document.getElementById(`${type}-total-cost`);
        
        if (costEl) costEl.textContent = `$${estimatedCost.toFixed(2)}`;
        if (commissionEl) commissionEl.textContent = `$${commission.toFixed(2)}`;
        if (totalEl) totalEl.textContent = `$${Math.abs(total).toFixed(2)}`;
        
        // Special handling for futures margin
        if (type === 'futures') {
            const marginEl = document.getElementById('futures-margin');
            if (marginEl) {
                const margin = estimatedCost * 0.1; // 10% margin requirement
                marginEl.textContent = `$${margin.toFixed(2)}`;
            }
        }
        
        return total;
    },
    
    // Submit order
    async submitOrder(type) {
        const form = document.getElementById(`${type}-order-form`);
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        const symbolInput = document.getElementById(`${type}-symbol`);
        const quantityInput = document.getElementById(`${type}-quantity`);
        const orderTypeSelect = document.getElementById(`${type}-order-type`);
        const priceInput = document.getElementById(`${type}-price`);
        const tifSelect = document.getElementById(`${type}-tif`);
        const sideBtn = document.querySelector(`#${type} .side-btn.active`);
        const submitBtn = document.getElementById(`${type}-submit-btn`);
        
        const orderData = {
            symbol: symbolInput.value,
            quantity: parseFloat(quantityInput.value),
            side: sideBtn?.dataset.side || 'buy',
            type: orderTypeSelect.value,
            time_in_force: tifSelect.value,
            asset_type: type
        };
        
        if (orderData.type === 'limit' || orderData.type === 'stop') {
            orderData.limit_price = parseFloat(priceInput.value);
        }
        
        // Validate buying power
        const totalCost = this.calculateOrderValue(type);
        const buyingPower = this.getBuyingPower(type);
        
        if (orderData.side === 'buy' && totalCost > buyingPower) {
            this.showMessage(`Insufficient buying power. Required: $${totalCost.toFixed(2)}, Available: $${buyingPower.toFixed(2)}`, 'error');
            return;
        }
        
        // Show confirmation dialog
        const confirmed = await this.showOrderConfirmation(orderData, totalCost);
        if (!confirmed) return;
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Placing Order...';
        
        try {
            const response = await fetch('/api/orders', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(orderData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showMessage(`Order placed successfully! Order ID: ${result.order_id || 'MOCK-' + Date.now()}`, 'success');
                
                // Clear form
                form.reset();
                
                // Hide symbol info
                const infoEl = document.getElementById(`${type}-symbol-info`);
                if (infoEl) infoEl.style.display = 'none';
                
                // Reload orders
                await this.loadOrders(type);
                
                // Update buying power
                await this.updateBuyingPower();
            } else {
                this.showMessage(result.error || 'Failed to place order', 'error');
            }
        } catch (error) {
            console.error('Order submission error:', error);
            this.showMessage('Failed to submit order. Please try again.', 'error');
        } finally {
            // Restore button
            submitBtn.disabled = false;
            const btnText = submitBtn.querySelector('.btn-text');
            if (!btnText) {
                submitBtn.innerHTML = `<i class="fas fa-paper-plane"></i> <span class="btn-text">Place ${orderData.side === 'buy' ? 'Buy' : 'Sell'} Order</span>`;
            } else {
                submitBtn.innerHTML = `<i class="fas fa-paper-plane"></i> <span class="btn-text">Place ${orderData.side === 'buy' ? 'Buy' : 'Sell'} Order</span>`;
            }
        }
    },
    
    // Show order confirmation dialog
    async showOrderConfirmation(orderData, totalCost) {
        const message = `
            <div class="order-confirmation">
                <h3>Confirm Order</h3>
                <div class="confirmation-details">
                    <div><strong>Action:</strong> ${orderData.side.toUpperCase()}</div>
                    <div><strong>Symbol:</strong> ${orderData.symbol}</div>
                    <div><strong>Quantity:</strong> ${orderData.quantity}</div>
                    <div><strong>Order Type:</strong> ${orderData.type.toUpperCase()}</div>
                    ${orderData.limit_price ? `<div><strong>Limit Price:</strong> $${orderData.limit_price.toFixed(2)}</div>` : ''}
                    <div><strong>Time in Force:</strong> ${orderData.time_in_force.toUpperCase()}</div>
                    <div class="total-cost"><strong>Total Cost:</strong> $${Math.abs(totalCost).toFixed(2)}</div>
                </div>
                <p>Are you sure you want to place this order?</p>
            </div>
        `;
        
        // For now, use confirm dialog. In production, use a modal
        return confirm(message.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' '));
    },
    
    // Load trading interface
    async loadTradingInterface(type) {
        console.log(`📈 Loading ${type} trading interface...`);
        
        // Start market data updates
        this.startMarketDataUpdates(type);
        
        // Load initial data
        await Promise.all([
            this.loadWatchlist(type),
            this.loadOrders(type),
            this.loadMarketMovers(type),
            this.updateBuyingPower()
        ]);
    },
    
    // Start market data updates
    startMarketDataUpdates(type) {
        // Clear existing interval
        if (this.marketDataInterval) {
            clearInterval(this.marketDataInterval);
        }
        
        // Update every 5 seconds
        this.marketDataInterval = setInterval(async () => {
            const section = document.getElementById(type);
            if (section && section.classList.contains('active')) {
                await this.updateMarketData(type);
            } else {
                // Stop updates if section is not active
                clearInterval(this.marketDataInterval);
                this.marketDataInterval = null;
            }
        }, 5000);
        
        // Initial update
        this.updateMarketData(type);
    },
    
    // Update market data
    async updateMarketData(type) {
        try {
            // Update watchlist prices
            const watchlistSymbols = this.watchlists[type] || [];
            if (watchlistSymbols.length > 0) {
                const response = await fetch(`/api/market/prices?symbols=${watchlistSymbols.join(',')}`);
                const data = await response.json();
                
                if (data.prices) {
                    this.updateWatchlistPrices(type, data.prices);
                }
            }
            
            // Update popular/top items
            if (type === 'stocks') {
                await this.updateMarketMovers(type);
            } else if (type === 'futures') {
                await this.updatePopularFutures();
            } else if (type === 'crypto') {
                await this.updateTopCrypto();
            }
        } catch (error) {
            console.error('Market data update error:', error);
        }
    },
    
    // Load watchlist
    async loadWatchlist(type) {
        const loadingEl = document.getElementById(`${type}-watchlist-loading`);
        const contentEl = document.getElementById(`${type}-watchlist-content`);
        const itemsEl = document.getElementById(`${type}-watchlist-items`);
        
        if (loadingEl) loadingEl.style.display = 'block';
        if (contentEl) contentEl.style.display = 'none';
        
        try {
            const response = await fetch(`/api/market/watchlist?type=${type}`);
            const data = await response.json();
            
            if (data.watchlist) {
                this.watchlists[type] = data.watchlist;
                this.renderWatchlist(type, data.watchlist);
            } else {
                // Use mock data
                const mockWatchlist = type === 'stocks' ? ['AAPL', 'GOOGL', 'MSFT'] :
                                     type === 'futures' ? ['ES', 'NQ', 'GC'] :
                                     ['BTC-USD', 'ETH-USD', 'ADA-USD'];
                this.watchlists[type] = mockWatchlist;
                this.renderWatchlist(type, mockWatchlist);
            }
            
            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
        } catch (error) {
            console.error('Watchlist loading error:', error);
            // Use mock data
            const mockWatchlist = type === 'stocks' ? ['AAPL', 'GOOGL', 'MSFT'] :
                                 type === 'futures' ? ['ES', 'NQ', 'GC'] :
                                 ['BTC-USD', 'ETH-USD', 'ADA-USD'];
            this.watchlists[type] = mockWatchlist;
            this.renderWatchlist(type, mockWatchlist);
            
            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
        }
    },
    
    // Render watchlist
    renderWatchlist(type, symbols) {
        const itemsEl = document.getElementById(`${type}-watchlist-items`);
        if (!itemsEl) return;
        
        itemsEl.innerHTML = '';
        
        symbols.forEach(symbol => {
            const item = document.createElement('div');
            item.className = 'watchlist-item';
            item.dataset.symbol = symbol;
            item.innerHTML = `
                <div class="watchlist-info">
                    <span class="watchlist-symbol">${symbol}</span>
                    <span class="watchlist-price">$${(100 + Math.random() * 50).toFixed(2)}</span>
                    <span class="watchlist-change positive">+${(Math.random() * 5).toFixed(2)}%</span>
                </div>
                <div class="watchlist-actions">
                    <button class="quick-trade-btn buy" data-symbol="${symbol}" data-side="buy">Buy</button>
                    <button class="quick-trade-btn sell" data-symbol="${symbol}" data-side="sell">Sell</button>
                    <button class="remove-watchlist-btn" data-symbol="${symbol}">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            itemsEl.appendChild(item);
        });
        
        // Setup event listeners for quick trade buttons
        itemsEl.querySelectorAll('.quick-trade-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const symbol = e.target.dataset.symbol;
                const side = e.target.dataset.side;
                this.quickTrade(type, symbol, side);
            });
        });
        
        // Setup remove buttons
        itemsEl.querySelectorAll('.remove-watchlist-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const symbol = e.target.dataset.symbol;
                this.removeFromWatchlist(type, symbol);
            });
        });
    },
    
    // Quick trade
    quickTrade(type, symbol, side) {
        // Pre-fill the order form
        const symbolInput = document.getElementById(`${type}-symbol`);
        const sideBtn = document.querySelector(`#${type} .side-btn[data-side="${side}"]`);
        
        if (symbolInput) {
            symbolInput.value = symbol;
            this.loadSymbolInfo(type, symbol);
        }
        
        if (sideBtn) {
            sideBtn.click();
        }
        
        // Focus on quantity input
        const quantityInput = document.getElementById(`${type}-quantity`);
        if (quantityInput) {
            quantityInput.focus();
        }
    },
    
    // Remove from watchlist
    async removeFromWatchlist(type, symbol) {
        try {
            await fetch(`/api/market/watchlist/${symbol}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type })
            });
            
            // Remove from local cache
            this.watchlists[type] = this.watchlists[type].filter(s => s !== symbol);
            
            // Re-render watchlist
            this.renderWatchlist(type, this.watchlists[type]);
            
            this.showMessage(`${symbol} removed from watchlist`, 'success');
        } catch (error) {
            console.error('Remove from watchlist error:', error);
            
            // Remove locally anyway
            this.watchlists[type] = this.watchlists[type].filter(s => s !== symbol);
            this.renderWatchlist(type, this.watchlists[type]);
        }
    },
    
    // Show add to watchlist dialog
    async showAddToWatchlist(type) {
        const symbol = prompt(`Enter ${type} symbol to add to watchlist:`);
        if (!symbol) return;
        
        try {
            await fetch('/api/market/watchlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol.toUpperCase(), type })
            });
            
            // Add to local cache
            if (!this.watchlists[type].includes(symbol.toUpperCase())) {
                this.watchlists[type].push(symbol.toUpperCase());
            }
            
            // Re-render watchlist
            this.renderWatchlist(type, this.watchlists[type]);
            
            this.showMessage(`${symbol.toUpperCase()} added to watchlist`, 'success');
        } catch (error) {
            console.error('Add to watchlist error:', error);
            
            // Add locally anyway
            if (!this.watchlists[type].includes(symbol.toUpperCase())) {
                this.watchlists[type].push(symbol.toUpperCase());
            }
            this.renderWatchlist(type, this.watchlists[type]);
        }
    },
    
    // Load orders
    async loadOrders(type) {
        const loadingEl = document.getElementById(`${type}-orders-loading`);
        const contentEl = document.getElementById(`${type}-orders-content`);
        
        if (loadingEl) loadingEl.style.display = 'block';
        if (contentEl) contentEl.style.display = 'none';
        
        try {
            const response = await fetch(`/api/orders?type=${type}`);
            const data = await response.json();
            
            if (data.orders) {
                this.activeOrders[type] = data.orders;
                this.renderOrders(type, data.orders);
            } else {
                // Use mock data
                const mockOrders = this.generateMockOrders(type);
                this.activeOrders[type] = mockOrders;
                this.renderOrders(type, mockOrders);
            }
            
            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
        } catch (error) {
            console.error('Orders loading error:', error);
            // Use mock data
            const mockOrders = this.generateMockOrders(type);
            this.activeOrders[type] = mockOrders;
            this.renderOrders(type, mockOrders);
            
            if (loadingEl) loadingEl.style.display = 'none';
            if (contentEl) contentEl.style.display = 'block';
        }
    },
    
    // Generate mock orders
    generateMockOrders(type) {
        const symbols = type === 'stocks' ? ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] :
                       type === 'futures' ? ['ES', 'NQ', 'GC', 'CL'] :
                       ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD'];
        
        return Array.from({ length: 5 }, (_, i) => ({
            id: `ORDER-${Date.now()}-${i}`,
            symbol: symbols[i % symbols.length],
            side: i % 2 === 0 ? 'buy' : 'sell',
            type: ['market', 'limit'][i % 2],
            quantity: Math.floor(Math.random() * 100) + 1,
            price: (100 + Math.random() * 50).toFixed(2),
            status: ['filled', 'pending', 'cancelled'][i % 3],
            created_at: new Date(Date.now() - i * 3600000).toISOString()
        }));
    },
    
    // Render orders
    renderOrders(type, orders) {
        const tbody = document.getElementById(`${type}-orders-table-body`);
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        orders.forEach(order => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${order.symbol}</td>
                <td><span class="order-side ${order.side}">${order.side.toUpperCase()}</span></td>
                <td>${order.type.toUpperCase()}</td>
                <td>${order.quantity}</td>
                <td>$${order.price}</td>
                <td><span class="order-status ${order.status}">${order.status.toUpperCase()}</span></td>
                <td>${new Date(order.created_at).toLocaleString()}</td>
                <td>
                    ${order.status === 'pending' ? 
                        `<button class="cancel-order-btn" data-order-id="${order.id}">Cancel</button>` :
                        `<button class="reorder-btn" data-order='${JSON.stringify(order)}'>Reorder</button>`
                    }
                </td>
            `;
            tbody.appendChild(row);
        });
        
        // Setup cancel buttons
        tbody.querySelectorAll('.cancel-order-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const orderId = e.target.dataset.orderId;
                this.cancelOrder(type, orderId);
            });
        });
        
        // Setup reorder buttons
        tbody.querySelectorAll('.reorder-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const order = JSON.parse(e.target.dataset.order);
                this.reorder(type, order);
            });
        });
    },
    
    // Cancel order
    async cancelOrder(type, orderId) {
        if (!confirm('Are you sure you want to cancel this order?')) return;
        
        try {
            const response = await fetch(`/api/orders/${orderId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.showMessage('Order cancelled successfully', 'success');
                await this.loadOrders(type);
            } else {
                this.showMessage('Failed to cancel order', 'error');
            }
        } catch (error) {
            console.error('Cancel order error:', error);
            this.showMessage('Failed to cancel order', 'error');
        }
    },
    
    // Reorder
    reorder(type, order) {
        // Pre-fill the order form with previous order details
        const symbolInput = document.getElementById(`${type}-symbol`);
        const quantityInput = document.getElementById(`${type}-quantity`);
        const orderTypeSelect = document.getElementById(`${type}-order-type`);
        const priceInput = document.getElementById(`${type}-price`);
        const sideBtn = document.querySelector(`#${type} .side-btn[data-side="${order.side}"]`);
        
        if (symbolInput) symbolInput.value = order.symbol;
        if (quantityInput) quantityInput.value = order.quantity;
        if (orderTypeSelect) {
            orderTypeSelect.value = order.type;
            orderTypeSelect.dispatchEvent(new Event('change'));
        }
        if (priceInput && order.type === 'limit') priceInput.value = order.price;
        if (sideBtn) sideBtn.click();
        
        // Scroll to form
        document.getElementById(`${type}-order-form`)?.scrollIntoView({ behavior: 'smooth' });
    },
    
    // Filter orders by status
    filterOrders(type, status) {
        const orders = this.activeOrders[type] || [];
        const filtered = status ? orders.filter(o => o.status === status) : orders;
        this.renderOrders(type, filtered);
    },
    
    // Filter orders by date
    filterOrdersByDate(type, date) {
        if (!date) {
            this.renderOrders(type, this.activeOrders[type] || []);
            return;
        }
        
        const orders = this.activeOrders[type] || [];
        const filtered = orders.filter(o => {
            const orderDate = new Date(o.created_at).toISOString().split('T')[0];
            return orderDate === date;
        });
        this.renderOrders(type, filtered);
    },
    
    // Load market movers (stocks)
    async loadMarketMovers(type) {
        if (type !== 'stocks') return;
        
        try {
            const response = await fetch('/api/market/movers');
            const data = await response.json();
            
            if (data.gainers && data.losers) {
                this.renderMarketMovers(data.gainers, data.losers);
            } else {
                // Mock data
                this.renderMockMarketMovers();
            }
        } catch (error) {
            console.error('Market movers error:', error);
            this.renderMockMarketMovers();
        }
    },
    
    // Render market movers
    renderMarketMovers(gainers, losers) {
        const gainersEl = document.getElementById('stocks-gainers');
        const losersEl = document.getElementById('stocks-losers');
        
        if (gainersEl) {
            gainersEl.innerHTML = gainers.map(stock => `
                <div class="mover-item">
                    <span class="mover-symbol">${stock.symbol}</span>
                    <span class="mover-price">$${stock.price.toFixed(2)}</span>
                    <span class="mover-change positive">+${stock.changePercent.toFixed(2)}%</span>
                </div>
            `).join('');
        }
        
        if (losersEl) {
            losersEl.innerHTML = losers.map(stock => `
                <div class="mover-item">
                    <span class="mover-symbol">${stock.symbol}</span>
                    <span class="mover-price">$${stock.price.toFixed(2)}</span>
                    <span class="mover-change negative">${stock.changePercent.toFixed(2)}%</span>
                </div>
            `).join('');
        }
    },
    
    // Render mock market movers
    renderMockMarketMovers() {
        const mockGainers = [
            { symbol: 'NVDA', price: 450.25, changePercent: 5.2 },
            { symbol: 'TSLA', price: 185.50, changePercent: 3.8 },
            { symbol: 'META', price: 320.75, changePercent: 2.9 }
        ];
        
        const mockLosers = [
            { symbol: 'BA', price: 205.30, changePercent: -3.5 },
            { symbol: 'DIS', price: 90.15, changePercent: -2.8 },
            { symbol: 'NKE', price: 95.40, changePercent: -1.9 }
        ];
        
        this.renderMarketMovers(mockGainers, mockLosers);
    },
    
    // Update popular futures prices
    async updatePopularFutures() {
        const contracts = document.querySelectorAll('#futures .contract-item');
        
        contracts.forEach(item => {
            const priceEl = item.querySelector('.price');
            const changeEl = item.querySelector('.change');
            
            // Mock price update
            const price = 4500 + Math.random() * 500;
            const change = (Math.random() - 0.5) * 5;
            
            if (priceEl) priceEl.textContent = `$${price.toFixed(2)}`;
            if (changeEl) {
                changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeEl.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
            }
        });
    },
    
    // Update top crypto prices
    async updateTopCrypto() {
        const cryptoItems = document.querySelectorAll('#crypto .crypto-item');
        
        cryptoItems.forEach(item => {
            const priceEl = item.querySelector('.price');
            const changeEl = item.querySelector('.change');
            
            // Mock price update
            const symbol = item.dataset.symbol;
            const basePrice = symbol === 'BTC-USD' ? 45000 : symbol === 'ETH-USD' ? 2500 : 100;
            const price = basePrice + Math.random() * (basePrice * 0.1);
            const change = (Math.random() - 0.5) * 10;
            
            if (priceEl) priceEl.textContent = `$${price.toFixed(2)}`;
            if (changeEl) {
                changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeEl.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
            }
        });
    },
    
    // Update market movers
    async updateMarketMovers(type) {
        await this.loadMarketMovers(type);
    },
    
    // Update watchlist prices
    updateWatchlistPrices(type, prices) {
        const items = document.querySelectorAll(`#${type}-watchlist-items .watchlist-item`);
        
        items.forEach(item => {
            const symbol = item.dataset.symbol;
            if (prices[symbol]) {
                const priceEl = item.querySelector('.watchlist-price');
                const changeEl = item.querySelector('.watchlist-change');
                
                if (priceEl) priceEl.textContent = `$${prices[symbol].last.toFixed(2)}`;
                if (changeEl) {
                    const change = prices[symbol].changePercent;
                    changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                    changeEl.className = `watchlist-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
            }
        });
    },
    
    // Update buying power
    async updateBuyingPower() {
        try {
            const response = await fetch('/api/portfolio/buying_power');
            const data = await response.json();
            
            if (data.buying_power) {
                ['stocks', 'futures', 'crypto'].forEach(type => {
                    const el = document.getElementById(`${type}-buying-power`);
                    if (el) el.textContent = `$${data.buying_power[type]?.toFixed(2) || data.buying_power.toFixed(2)}`;
                });
            } else {
                // Mock data
                ['stocks', 'futures', 'crypto'].forEach(type => {
                    const el = document.getElementById(`${type}-buying-power`);
                    if (el) el.textContent = '$10,000.00';
                });
            }
        } catch (error) {
            console.error('Buying power update error:', error);
            // Mock data
            ['stocks', 'futures', 'crypto'].forEach(type => {
                const el = document.getElementById(`${type}-buying-power`);
                if (el) el.textContent = '$10,000.00';
            });
        }
    },
    
    // Get buying power
    getBuyingPower(type) {
        const el = document.getElementById(`${type}-buying-power`);
        if (el) {
            return parseFloat(el.textContent.replace('$', '').replace(',', ''));
        }
        return 10000; // Default
    },
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Check if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            const activeSection = document.querySelector('.content-section.active');
            if (!activeSection) return;
            
            const type = activeSection.id;
            if (!['stocks', 'futures', 'crypto'].includes(type)) return;
            
            // B key - Buy
            if (e.key === 'b' || e.key === 'B') {
                e.preventDefault();
                const buyBtn = document.querySelector(`#${type} .side-btn[data-side="buy"]`);
                if (buyBtn) buyBtn.click();
            }
            
            // S key - Sell
            if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                const sellBtn = document.querySelector(`#${type} .side-btn[data-side="sell"]`);
                if (sellBtn) sellBtn.click();
            }
            
            // Enter key - Submit order
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                const submitBtn = document.getElementById(`${type}-submit-btn`);
                if (submitBtn && !submitBtn.disabled) submitBtn.click();
            }
        });
    },
    
    // Load trading signals
    async loadTradingSignals() {
        const signalsSection = document.getElementById('signals');
        if (!signalsSection) return;
        
        signalsSection.innerHTML = `
            <div class="section-header">
                <h2><i class="fas fa-signal"></i> Trading Signals</h2>
            </div>
            <div class="signals-grid">
                ${this.generateMockSignals()}
            </div>
        `;
    },
    
    // Generate mock trading signals
    generateMockSignals() {
        const signals = [
            { symbol: 'AAPL', type: 'BUY', strength: 'Strong', price: 175.50, reason: 'RSI Oversold' },
            { symbol: 'GOOGL', type: 'SELL', strength: 'Moderate', price: 140.25, reason: 'Resistance Hit' },
            { symbol: 'MSFT', type: 'BUY', strength: 'Strong', price: 380.75, reason: 'Breakout Pattern' },
            { symbol: 'BTC-USD', type: 'HOLD', strength: 'Weak', price: 45000, reason: 'Consolidation' }
        ];
        
        return signals.map(signal => `
            <div class="signal-card ${signal.type.toLowerCase()}">
                <div class="signal-header">
                    <span class="signal-symbol">${signal.symbol}</span>
                    <span class="signal-type ${signal.type.toLowerCase()}">${signal.type}</span>
                </div>
                <div class="signal-details">
                    <div class="signal-price">$${signal.price.toFixed(2)}</div>
                    <div class="signal-strength">Strength: ${signal.strength}</div>
                    <div class="signal-reason">${signal.reason}</div>
                </div>
            </div>
        `).join('');
    }
});

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}