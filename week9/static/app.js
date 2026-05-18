/**
 * Classical Mechanics Solver - Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentTopic = 'euler_rk4';
    let isLoading = false;

    // Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const exampleBtns = document.querySelectorAll('.example-btn');
    const problemInput = document.getElementById('problem-input');
    const solveBtn = document.getElementById('solve-btn');
    const btnText = solveBtn.querySelector('.btn-text');
    const btnSpinner = solveBtn.querySelector('.btn-spinner');
    const statusBar = document.getElementById('status-bar');
    const chartContainer = document.getElementById('chart-container');
    const explanationContent = document.getElementById('explanation-content');
    const paramsContent = document.getElementById('params-content');
    const explanationSection = document.getElementById('explanation-section');
    
    const apiBanner = document.getElementById('api-banner');
    const apiBadge = document.getElementById('api-badge');
    const apiKeySection = document.getElementById('api-key-section');
    const apiKeyInput = document.getElementById('api-key-input');
    const saveApiKeyBtn = document.getElementById('save-api-key-btn');

    // Tab Handlers
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const topic = btn.getAttribute('data-topic');
            if (topic === currentTopic) return;

            // Update UI
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTopic = topic;

            updateExamples(topic);
        });
    });

    // Example Button Handlers
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const exampleText = btn.getAttribute('data-example');
            problemInput.value = exampleText;
        });
    });

    // Solve Button Handler
    solveBtn.addEventListener('click', solveProblem);

    // Save API Key Handler
    saveApiKeyBtn.addEventListener('click', async () => {
        const key = apiKeyInput.value.trim();
        if (!key) return;

        try {
            const response = await fetch('/set-api-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: key })
            });
            if (response.ok) {
                apiKeySection.style.display = 'none';
                apiBanner.style.display = 'none';
                solveProblem();
            }
        } catch (error) {
            console.error('Failed to set API key:', error);
        }
    });

    function updateExamples(topic) {
        exampleBtns.forEach(btn => {
            if (btn.getAttribute('data-topic') === topic) {
                btn.removeAttribute('hidden');
                btn.style.display = 'inline-block';
            } else {
                btn.setAttribute('hidden', '');
                btn.style.display = 'none';
            }
        });
    }

    async function solveProblem() {
        const problem = problemInput.value.trim();
        if (!problem) {
            showStatus('문제를 입력해주세요', 'error');
            return;
        }

        if (isLoading) return;

        setLoading(true);
        showStatus('AI가 파라미터를 추출하고 계산 중...', 'success');
        apiBadge.style.display = 'none';
        apiBanner.style.display = 'none';

        try {
            const response = await fetch('/solve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    problem: problem,
                    topic: currentTopic
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                showStatus(`오류: ${result.error}`, 'error');
            } else {
                showStatus('계산 완료!', 'success');
                renderChart(result.plotly_json);
                renderExplanation(result.explanation);
                renderParams(result.parameters, result.steps);
                handleApiFeedback(result);
            }

        } catch (error) {
            console.error('Solve failed:', error);
            showStatus(`통신 오류가 발생했습니다: ${error.message}`, 'error');
        } finally {
            setLoading(false);
        }
    }

    function handleApiFeedback(result) {
        if (result.api_needed && !result.api_used) {
            apiBanner.textContent = '입력이 모호하여 기본값으로 계산했습니다. API 키를 입력하면 자연어를 정확히 해석합니다.';
            apiBanner.className = 'banner warning';
            apiBanner.style.display = 'block';
            apiKeySection.style.display = 'block';
        } else if (result.api_used) {
            apiBadge.textContent = 'Claude AI 사용됨';
            apiBadge.className = 'badge success';
            apiBadge.style.display = 'inline-block';
        } else if (!result.api_needed) {
            apiBadge.textContent = 'API 없이 계산됨';
            apiBadge.className = 'badge info';
            apiBadge.style.display = 'inline-block';
        }
    }

    function setLoading(loading) {
        isLoading = loading;
        solveBtn.disabled = loading;
        if (loading) {
            btnText.setAttribute('hidden', '');
            btnSpinner.removeAttribute('hidden');
        } else {
            btnText.removeAttribute('hidden');
            btnSpinner.setAttribute('hidden', '');
        }
    }

    function showStatus(message, type) {
        statusBar.textContent = message;
        statusBar.className = `status-bar ${type}`;
        statusBar.removeAttribute('hidden');
    }

    function renderChart(plotlyJson) {
        if (!plotlyJson) return;

        // Apply dark theme styles to layout
        const layout = {
            ...plotlyJson.layout,
            paper_bgcolor: '#0d1117',
            plot_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            margin: { t: 40, b: 40, l: 60, r: 20 },
            xaxis: {
                ...(plotlyJson.layout?.xaxis || {}),
                gridcolor: '#30363d',
                linecolor: '#30363d',
                zerolinecolor: '#30363d'
            },
            yaxis: {
                ...(plotlyJson.layout?.yaxis || {}),
                gridcolor: '#30363d',
                linecolor: '#30363d',
                zerolinecolor: '#30363d'
            }
        };

        // If it's a 3D plot, adjust scene
        if (layout.scene) {
            layout.scene.xaxis.gridcolor = '#30363d';
            layout.scene.yaxis.gridcolor = '#30363d';
            layout.scene.zaxis.gridcolor = '#30363d';
            layout.scene.xaxis.backgroundcolor = '#0d1117';
            layout.scene.yaxis.backgroundcolor = '#0d1117';
            layout.scene.zaxis.backgroundcolor = '#0d1117';
        }

        // Clear placeholder if it exists
        const placeholder = chartContainer.querySelector('.chart-placeholder');
        if (placeholder) {
            chartContainer.innerHTML = '';
        }

        Plotly.newPlot('chart-container', plotlyJson.data, layout, { responsive: true });
    }

    function renderExplanation(markdown) {
        if (!markdown) {
            explanationContent.innerHTML = '해설을 불러오지 못했습니다.';
            return;
        }
        explanationContent.innerHTML = marked.parse(markdown);
        explanationSection.open = true;
    }

    function renderParams(params, steps) {
        let html = '<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>';
        
        if (params && typeof params === 'object') {
            for (const [key, value] of Object.entries(params)) {
                html += `<tr><td>${key}</td><td>${value}</td></tr>`;
            }
        } else {
            html += '<tr><td colspan="2">추출된 파라미터가 없습니다.</td></tr>';
        }
        
        html += '</tbody></table>';

        if (steps && Array.isArray(steps)) {
            html += '<h4 style="margin-top: 12px; margin-bottom: 6px; color: #8b949e;">Calculation Steps</h4>';
            html += '<ul style="padding-left: 20px;">';
            steps.forEach(step => {
                html += `<li>${step}</li>`;
            });
            html += '</ul>';
        }

        paramsContent.innerHTML = html;
    }

    // Initialize examples
    updateExamples(currentTopic);
});
