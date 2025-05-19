def create_connectivity_html(input_tokens, output_tokens, model_name, attn_weights, top3_predictions=None, threshold=0.1):
    """
    Create an HTML representation of the connectivity between input and output tokens
    with improved support for Devanagari script and top 3 token predictions on hover.
    
    Args:
        input_tokens: List of input characters
        output_tokens: List of output characters 
        attn_weights: Attention matrix of shape [output_len, input_len]
        top3_predictions: List of lists containing top 3 predictions for each output position
                         [[(token1, prob1), (token2, prob2), (token3, prob3)], ...]
        threshold: Minimum attention weight to show connection
        
    Returns:
        HTML string representation of the connectivity visualization
    """
    # Clean input and output tokens (remove padding, BOS, EOS)
    clean_input = [t for t in input_tokens if t not in ['<pad>', '<bos>', '<eos>', '?']]
    clean_output = [t for t in output_tokens if t not in ['<pad>', '<bos>', '<eos>', '?']]
    
    # Resize attention weights matrix to match cleaned tokens
    clean_attn = attn_weights[:len(clean_output), :len(clean_input)]
    
    # If no top3 predictions provided, create dummy data
    if top3_predictions is None:
        top3_predictions = [[("—", 0.0), ("—", 0.0), ("—", 0.0)] for _ in range(len(clean_output))]
    
    # Generate HTML for visualization
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
            
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
            }}
            
            .container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                max-width: 1024px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .visualization {{
                display: flex;
                flex-direction: column;
                margin: 20px 0;
                position: relative;
                width: 100%;
            }}
            
            .row {{
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
                position: relative;
                width: 100%;
                z-index: 2;
            }}
            
            .input-row, .output-row {{
                display: flex;
                justify-content: space-around;
                width: 100%;
            }}
            
            .cell {{
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid #ddd;
                background-color: #f8f8f8;
                margin: 0 5px;
                position: relative;
                font-size: 18px;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s ease-in-out;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            .input-cell {{
                font-family: 'Arial', sans-serif;
            }}
            
            .output-cell {{
                font-family: 'Noto Sans Devanagari', sans-serif;
                font-size: 20px;
            }}
            
            .connection-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }}
            
            .connection {{
                position: absolute;
                background-color: rgba(0, 128, 0, 0.5);
                height: 2px;
                transform-origin: 0 0;
                pointer-events: none;
                transition: opacity 0.3s ease, background-color 0.3s ease, height 0.3s ease;
            }}
            
            .model-type {{
                font-weight: bold;
                font-size: 20px;
                margin: 10px;
                background-color: #eee;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            .highlighted {{
                background-color: #ffeb3b;
                box-shadow: 0 0 8px rgba(0,0,0,0.3);
                transform: scale(1.1);
                z-index: 10;
            }}
            
            .title {{
                font-weight: bold;
                font-size: 24px;
                margin: 20px 0;
                color: #333;
            }}
            
            .instructions {{
                margin: 15px 0;
                font-style: italic;
                color: #666;
                text-align: center;
                max-width: 600px;
                line-height: 1.5;
            }}
            
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 15px 0;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            .slider-container {{
                display: flex;
                align-items: center;
                margin-right: 20px;
            }}
            
            .slider-label {{
                margin-right: 10px;
                font-weight: bold;
            }}
            
            #threshold-slider {{
                width: 150px;
                -webkit-appearance: none;
                height: 6px;
                border-radius: 3px;
                background: #d3d3d3;
                outline: none;
            }}
            
            #threshold-slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%; 
                background: #4CAF50;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            #threshold-slider::-webkit-slider-thumb:hover {{
                transform: scale(1.2);
            }}
            
            #threshold-value {{
                margin-left: 10px;
                min-width: 40px;
                text-align: center;
                font-weight: bold;
                color: #4CAF50;
            }}
            
            /* Tooltip for top-3 predictions */
            .prediction-tooltip {{
                display: none;
                position: absolute;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 100;
                width: 180px;
                top: 45px;
                left: 50%;
                transform: translateX(-50%);
                font-family: 'Noto Sans Devanagari', sans-serif;
                transition: opacity 0.3s ease;
                opacity: 0;
            }}
            
            .prediction-tooltip.visible {{
                display: block;
                opacity: 1;
            }}
            
            .prediction-tooltip::before {{
                content: '';
                position: absolute;
                top: -8px;
                left: 50%;
                transform: translateX(-50%);
                border-width: 0 8px 8px 8px;
                border-style: solid;
                border-color: transparent transparent #ddd transparent;
            }}
            
            .prediction-tooltip h4 {{
                margin: 0 0 8px 0;
                text-align: center;
                font-size: 14px;
                color: #444;
            }}
            
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                margin: 4px 0;
                padding: 3px;
                border-radius: 3px;
                transition: background-color 0.2s ease;
            }}
            
            .prediction-item:hover {{
                background-color: #f5f5f5;
            }}
            
            .prediction-char {{
                font-size: 18px;
                font-weight: bold;
            }}
            
            .prediction-prob {{
                font-size: 14px;
                color: #666;
            }}
            
            .prediction-prob-bar {{
                height: 6px;
                background-color: #4caf50;
                margin-top: 2px;
                border-radius: 3px;
                transition: width 0.3s ease;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">Transliteration Attention Connectivity</div>
            <div class="instructions">
                Hover over any character to see its connections. 
                The green lines show the attention weights: stronger connections have darker green lines.
                Hover over output characters to see top prediction alternatives.
                Use the threshold slider to filter the connections by strength.
            </div>
            
            <div class="controls">
                <div class="slider-container">
                    <span class="slider-label">Threshold:</span>
                    <input type="range" id="threshold-slider" min="0" max="1" step="0.05" value="0.1">
                    <span id="threshold-value">0.1</span>
                </div>
            </div>
            
            <div class="model-type">Sequence-to-Sequence with Attention [ {} ]</div>
            <div class="visualization">
                <div class="row input-row">
    """.format(model_name.upper())
    
    # Add input tokens
    for i, token in enumerate(clean_input):
        html += f'<div class="cell input-cell" data-index="{i}">{token}</div>'
    
    html += """
                </div>
                <div class="connection-container">
    """
    
    # Add connections between input and output tokens
    for i, out_token in enumerate(clean_output):
        for j, in_token in enumerate(clean_input):
            weight = float(clean_attn[i, j].item())
            if weight > threshold:
                # Calculate opacity based on weight (normalized between 0.1 and 1.0)
                opacity = max(0.1, min(1.0, weight))
                
                # We'll calculate the position and rotation with JavaScript
                html += f'<div class="connection" data-from-input="{j}" data-to-output="{i}" data-weight="{weight:.4f}" style="opacity: 0;"></div>'
    
    html += """
                </div>
                <div class="row output-row">
    """
    
    # Add output tokens with tooltips for predictions
    for i, token in enumerate(clean_output):
        # Get top 3 predictions for this position
        top3 = top3_predictions[i] if i < len(top3_predictions) else [("—", 0), ("—", 0), ("—", 0)]
        
        # Create tooltip HTML for predictions
        tooltip_html = f"""
        <div class="prediction-tooltip" id="tooltip-{i}">
            <h4>Top Predictions</h4>
        """
        
        # Add each prediction with probability bar
        for idx, (pred_token, prob) in enumerate(top3):
            bar_width = int(prob * 100)
            is_actual = pred_token == token
            highlight = 'background-color: #e6f7e6;' if is_actual else ''
            
            tooltip_html += f"""
            <div class="prediction-item" style="{highlight}">
                <div class="prediction-char">{pred_token}</div>
                <div class="prediction-prob">
                    <div>{prob:.2f}</div>
                    <div class="prediction-prob-bar" style="width: {bar_width}%;"></div>
                </div>
            </div>
            """
        
        tooltip_html += "</div>"
        
        # Add output cell with tooltip
        html += f"""
        <div class="cell output-cell" data-index="{i}">
            {token}
            {tooltip_html}
        </div>
        """
    
    html += """
                </div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                setupVisualization();
                window.addEventListener('resize', debounce(positionConnections, 100));
                
                const thresholdSlider = document.getElementById('threshold-slider');
                const thresholdValue = document.getElementById('threshold-value');
                
                thresholdSlider.addEventListener('input', function() {
                    const value = parseFloat(this.value);
                    thresholdValue.textContent = value.toFixed(2);
                    updateConnectionVisibility(value);
                });
            });
            
            // Debounce function to limit excessive calls during resize
            function debounce(func, wait) {
                let timeout;
                return function() {
                    const context = this;
                    const args = arguments;
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        func.apply(context, args);
                    }, wait);
                };
            }
            
            function setupVisualization() {
                positionConnections();
                
                // Add event listeners to cells for smooth interactions
                const inputCells = document.querySelectorAll('.input-cell');
                const outputCells = document.querySelectorAll('.output-cell');
                
                inputCells.forEach(cell => {
                    cell.addEventListener('mouseenter', function() {
                        const index = parseInt(this.getAttribute('data-index'));
                        highlightInputConnections(index);
                    });
                    
                    cell.addEventListener('mouseleave', function() {
                        resetHighlights();
                    });
                });
                
                outputCells.forEach(cell => {
                    cell.addEventListener('mouseenter', function() {
                        const index = parseInt(this.getAttribute('data-index'));
                        highlightOutputConnections(index);
                        showTooltip(index);
                    });
                    
                    cell.addEventListener('mouseleave', function() {
                        const index = parseInt(this.getAttribute('data-index'));
                        resetHighlights();
                        hideTooltip(index);
                    });
                });
            }

            function updateConnectionVisibility(threshold) {
                const connections = document.querySelectorAll('.connection');
                connections.forEach(conn => {
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    if (weight >= threshold) {
                        conn.style.display = 'block';
                        conn.style.opacity = weight;
                    } else {
                        conn.style.display = 'none';
                    }
                });
            }

            function positionConnections() {
                const connections = document.querySelectorAll('.connection');
                const inputCells = document.querySelectorAll('.input-cell');
                const outputCells = document.querySelectorAll('.output-cell');
                const containerRect = document.querySelector('.connection-container').getBoundingClientRect();
                
                connections.forEach(conn => {
                    const fromIndex = parseInt(conn.getAttribute('data-from-input'));
                    const toIndex = parseInt(conn.getAttribute('data-to-output'));
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    
                    if (fromIndex < inputCells.length && toIndex < outputCells.length) {
                        const fromCell = inputCells[fromIndex];
                        const toCell = outputCells[toIndex];
                        
                        const fromRect = fromCell.getBoundingClientRect();
                        const toRect = toCell.getBoundingClientRect();
                        
                        const fromX = fromRect.left + fromRect.width/2 - containerRect.left;
                        const fromY = fromRect.top + fromRect.height - containerRect.top;
                        const toX = toRect.left + toRect.width/2 - containerRect.left;
                        const toY = toRect.top - containerRect.top;
                        
                        const length = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
                        const angle = Math.atan2(toY - fromY, toX - fromX);
                        
                        conn.style.width = `${length}px`;
                        conn.style.left = `${fromX}px`;
                        conn.style.top = `${fromY}px`;
                        conn.style.transform = `rotate(${angle}rad)`;
                        
                        // Set default state
                        conn.style.zIndex = '1';
                        conn.style.opacity = conn.getAttribute('data-display') !== 'none' ? weight : 0;
                        conn.style.backgroundColor = `rgba(0, 128, 0, ${weight})`;
                    }
                });
                
                // Apply initial threshold
                const thresholdSlider = document.getElementById('threshold-slider');
                if (thresholdSlider) {
                    updateConnectionVisibility(parseFloat(thresholdSlider.value));
                }
            }

            function highlightInputConnections(index) {
                const connections = document.querySelectorAll(`.connection[data-from-input="${index}"]`);
                const cell = document.querySelector(`.input-cell[data-index="${index}"]`);
                
                cell.classList.add('highlighted');
                
                connections.forEach(conn => {
                    const toIndex = conn.getAttribute('data-to-output');
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    const thresholdValue = parseFloat(document.getElementById('threshold-slider').value);
                    
                    if (weight >= thresholdValue) {
                        document.querySelector(`.output-cell[data-index="${toIndex}"]`).classList.add('highlighted');
                        conn.style.height = '3px';
                        conn.style.zIndex = '5';
                        conn.style.backgroundColor = `rgba(76, 175, 80, ${weight})`;
                    }
                });
            }

            function highlightOutputConnections(index) {
                const connections = document.querySelectorAll(`.connection[data-to-output="${index}"]`);
                const cell = document.querySelector(`.output-cell[data-index="${index}"]`);
                
                cell.classList.add('highlighted');
                
                connections.forEach(conn => {
                    const fromIndex = conn.getAttribute('data-from-input');
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    const thresholdValue = parseFloat(document.getElementById('threshold-slider').value);
                    
                    if (weight >= thresholdValue) {
                        document.querySelector(`.input-cell[data-index="${fromIndex}"]`).classList.add('highlighted');
                        conn.style.height = '3px';
                        conn.style.zIndex = '5';
                        conn.style.backgroundColor = `rgba(76, 175, 80, ${weight})`;
                    }
                });
            }

            function resetHighlights() {
                document.querySelectorAll('.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });
                document.querySelectorAll('.connection').forEach(conn => {
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    const thresholdValue = parseFloat(document.getElementById('threshold-slider').value);
                    
                    if (weight >= thresholdValue) {
                        conn.style.height = '2px';
                        conn.style.zIndex = '1';
                        conn.style.backgroundColor = `rgba(0, 128, 0, ${weight})`;
                    }
                });
            }
            
            function showTooltip(index) {
                const tooltip = document.getElementById(`tooltip-${index}`);
                if (tooltip) {
                    tooltip.classList.add('visible');
                }
            }
            
            function hideTooltip(index) {
                const tooltip = document.getElementById(`tooltip-${index}`);
                if (tooltip) {
                    tooltip.classList.remove('visible');
                }
            }
        </script>
    </body>
    </html>
    """
    
    return html