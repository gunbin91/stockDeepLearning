// AI 주식 분석 시스템 - 메인 JavaScript

$(document).ready(function() {
    // 전역 변수
    let socket = null;
    let currentAnalysis = null;
    let currentBacktest = null;
    
    // 초기화
    initializeApp();
    
    function initializeApp() {
        // 플래그 초기화
        window.analysisRunning = false;
        window.backtestRunning = false;
        
        // WebSocket 연결
        connectWebSocket();
        
        // 공통 이벤트 리스너 설정
        setupCommonEventListeners();
        
        // 페이지별 초기화
        initializePageSpecific();
    }
    
    function connectWebSocket() {
        socket = io({
            timeout: 20000,        // 연결 타임아웃 20초
            pingTimeout: 60000,    // 핑 타임아웃 60초
            pingInterval: 25000    // 핑 간격 25초
        });
        
        socket.on('connect', function() {
            console.log('WebSocket 연결됨');
            // 초기 연결 시에만 토스트 표시
            if (!window.initialConnection) {
                showToast('서버에 연결되었습니다.', 'success');
                window.initialConnection = true;
            }
        });
        
        socket.on('disconnect', function() {
            console.log('WebSocket 연결 해제됨');
            // 재연결 시도 중이므로 토스트 표시하지 않음
        });
        
        socket.on('connect_error', function(error) {
            console.error('WebSocket 연결 오류:', error);
            showToast('서버 연결에 실패했습니다.', 'danger');
        });
    }
    
    function setupCommonEventListeners() {
        // 다크 모드 토글
        $('#darkModeToggle').on('click', function() {
            $('body').toggleClass('dark-mode');
            
            // 버튼 텍스트 업데이트
            if ($('body').hasClass('dark-mode')) {
                $(this).text('라이트 모드');
                localStorage.setItem('darkMode', 'enabled');
            } else {
                $(this).text('다크 모드');
                localStorage.setItem('darkMode', 'disabled');
            }
        });
        
        // 다크 모드 설정 로드
        if (localStorage.getItem('darkMode') === 'enabled') {
            $('body').addClass('dark-mode');
            $('#darkModeToggle').text('라이트 모드');
        } else {
            $('#darkModeToggle').text('다크 모드');
        }
        
        // 토스트 알림 자동 숨김
        $('.toast').on('hidden.bs.toast', function() {
            $(this).remove();
        });
        
        // 폼 유효성 검사
        $('form').on('submit', function(e) {
            if (!validateForm($(this))) {
                e.preventDefault();
                return false;
            }
        });
        
        // 숫자 입력 필드 포맷팅
        $('input[type="number"]').on('blur', function() {
            formatNumberInput($(this));
        });
    }
    
    function initializePageSpecific() {
        const currentPage = window.location.pathname;
        
        switch(currentPage) {
            case '/':
                initializeIndexPage();
                break;
            case '/model_analysis':
                initializeModelAnalysisPage();
                break;
            case '/backtest':
                initializeBacktestPage();
                break;
        }
    }
    
    function initializeIndexPage() {
        // 주식 테이블 초기화
        if ($('#stock_table').length) {
            initializeStockTable();
        }
        
        // 분석 관련 이벤트
        setupAnalysisEvents();
    }
    
    function initializeModelAnalysisPage() {
        // 피처 중요도 차트 초기화
        if ($('#feature_importance_chart').length) {
            initializeFeatureChart();
        }
        
        // 피처 테이블 초기화
        if ($('#feature_table').length) {
            initializeFeatureTable();
        }
    }
    
    function initializeBacktestPage() {
        // 백테스팅 관련 이벤트
        setupBacktestEvents();
        
        // 기존 리포트 로드
        if ($('#backtest_report').length) {
            loadBacktestReport();
        }
    }
    
    function initializeStockTable() {
        // DataTables 재초기화 방지
        if (!$.fn.DataTable.isDataTable('#stock_table')) {
            $('#stock_table').DataTable({
                pageLength: 25,
                order: [[0, 'asc']],
                language: {
                    "lengthMenu": "페이지당 _MENU_ 개씩 보기",
                    "zeroRecords": "데이터가 없습니다",
                    "info": "_START_ - _END_ / _TOTAL_ 개",
                    "infoEmpty": "0 개",
                    "infoFiltered": "(전체 _MAX_ 개 중 필터링됨)",
                    "search": "검색:",
                    "paginate": {
                        "first": "처음",
                        "last": "마지막",
                        "next": "다음",
                        "previous": "이전"
                    }
                },
                columnDefs: [
                    { targets: [0], width: '80px' },
                    { targets: [1], width: '120px' },
                    { targets: [2], width: '100px' },
                    { targets: [3, 4], width: '120px', className: 'text-end' },
                    { targets: [5], width: '120px', className: 'text-end' },
                    { targets: [6, 7, 8], width: '100px', className: 'text-end' },
                    { targets: [9], width: '120px', className: 'text-end' }
                ],
                responsive: true,
                scrollX: true,
                drawCallback: function() {
                    // 테이블이 다시 그려질 때마다 색상 적용
                    applyChangeRateColors();
                    applyPriceColors();
                }
            });
        }
    }
    
    function applyChangeRateColors() {
        $('.change-cell').each(function() {
            const changeText = $(this).text();
            if (changeText.includes('+')) {
                $(this).addClass('text-danger fw-bold');
            } else if (changeText.includes('-')) {
                $(this).addClass('text-primary fw-bold');
            }
        });
    }
    
    function applyPriceColors() {
        $('.price-cell').each(function() {
            const row = $(this).closest('tr');
            const currentPrice = parseFloat($(this).text().replace(/,/g, ''));
            const basePrice = parseFloat(row.find('td:eq(5)').text().replace(/,/g, ''));
            
            if (currentPrice > basePrice) {
                $(this).addClass('text-danger fw-bold'); // 상승: 빨간색
            } else if (currentPrice < basePrice) {
                $(this).addClass('text-primary fw-bold'); // 하락: 파란색
            }
        });
    }
    
    function setupAnalysisEvents() {
        // 분석 시작 버튼
        $('#start_analysis_btn').on('click', function() {
            startAnalysis();
        });
        
        // 결과 초기화 버튼
        $('#clear_results_btn').on('click', function() {
            clearResults();
        });
        
        // 분석 중단 버튼 (팝업 내부)
        $('#stop_analysis_btn_modal').on('click', function() {
            stopAnalysis();
        });
        
        // 팝업이 닫힐 때 분석 종료
        $('#analysis_modal').on('hidden.bs.modal', function() {
            if (window.analysisRunning) {
                stopAnalysis();
            }
        });
        
        // 주식 행 클릭 (이벤트 위임으로 동적 생성된 행에도 적용)
        $(document).on('click', '.stock-row', function() {
            const ticker = $(this).data('ticker');
            const name = $(this).data('name');
            showStockDetails(ticker, name);
        });
        
        // 분석 로그 수신
        if (socket) {
            socket.on('analysis_log', function(data) {
                console.log('DEBUG: WebSocket 로그 수신 -', data.message);
                updateAnalysisLog(data.message);
            });
            
            socket.on('analysis_complete', function(data) {
                console.log('DEBUG: WebSocket 완료 수신 -', data);
                handleAnalysisComplete(data);
            });
        }
    }
    
    function startAnalysis() {
        const analysisDate = $('#analysis_date').val();
        if (!analysisDate) {
            showToast('분석 기준일을 선택해주세요.', 'warning');
            return;
        }
        
        // 이미 분석이 실행 중인지 확인
        if (window.analysisRunning) {
            showToast('이미 분석이 실행 중입니다.', 'warning');
            return;
        }
        
        // 버튼이 이미 비활성화되어 있는지 확인
        if ($('#start_analysis_btn').prop('disabled')) {
            showToast('이미 분석이 실행 중입니다.', 'warning');
            return;
        }
        
        // 1. 먼저 팝업 표시
        $('#analysis_modal').modal('show');
        
        // 2. 분석 실행 중 플래그 설정
        window.analysisRunning = true;
        
        // 3. 분석 시작 버튼 비활성화
        $('#start_analysis_btn').prop('disabled', true).text('분석 실행 중...');
        
        // 4. 분석 중단 버튼 표시 (팝업 내부)
        $('#stop_analysis_btn_modal').show();
        
        // 5. 분석 시작 요청
        $.ajax({
            url: '/api/start_analysis',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({analysis_date: analysisDate}),
            success: function(response) {
                console.log('분석 시작:', response);
                showToast('분석이 시작되었습니다.', 'info');
            },
            error: function(xhr) {
                // 분석 실행 중 플래그 해제
                window.analysisRunning = false;
                
                // 분석 시작 버튼 다시 활성화
                $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
                
                // 분석 중단 버튼 숨기기
                $('#stop_analysis_btn_modal').hide();
                
                const error = JSON.parse(xhr.responseText);
                showToast('분석 시작 중 오류: ' + error.error, 'danger');
                $('#analysis_modal').modal('hide');
            }
        });
    }
    
    function clearResults() {
        if (confirm('분석 결과를 초기화하시겠습니까?')) {
            location.reload();
        }
    }
    
    function stopAnalysis() {
        if (confirm('실행 중인 분석을 중단하시겠습니까?')) {
            $.ajax({
                url: '/api/stop_analysis',
                method: 'POST',
                success: function(response) {
                    showToast('분석이 중단되었습니다.', 'warning');
                    // 상태 복구
                    window.analysisRunning = false;
                    $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
                    $('#stop_analysis_btn_modal').hide();
                    $('#analysis_modal').modal('hide');
                },
                error: function(xhr) {
                    const error = JSON.parse(xhr.responseText);
                    showToast('분석 중단 중 오류: ' + error.error, 'danger');
                }
            });
        }
    }
    
    function showStockDetails(ticker, name) {
        // 종목코드를 문자열로 변환하고 6자리로 패딩 (호환성 개선)
        const tickerStr = String(ticker);
        const paddedTicker = tickerStr.length < 6 ? ('000000' + tickerStr).slice(-6) : tickerStr;
        
        // 차트 섹션 표시
        $('#chart_title').text(`📈 [${name}] 상세 차트`);
        $('#chart_section').show();
        
        // 피처 섹션 표시
        $('#features_title').text(`📊 ${name} (${ticker}) 분석 피처 데이터`);
        $('#features_section').show();
        
        // 차트 로딩
        $('#stock_chart').html('<div class="text-center"><i class="fas fa-spinner fa-spin fa-2x"></i><br>차트를 불러오는 중...</div>');
        
        // 차트 데이터 요청
        $.get(`/api/stock_chart/${paddedTicker}`)
            .done(function(data) {
                if (data.chart) {
                    try {
                        const chartData = JSON.parse(data.chart);
                        // 기존 차트 제거 후 새로 생성
                        $('#stock_chart').empty();
                        Plotly.newPlot('stock_chart', chartData.data, chartData.layout, {
                            responsive: true,
                            displayModeBar: true,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                        });
                        console.log('차트 렌더링 완료:', paddedTicker);
                    } catch (error) {
                        console.error('차트 렌더링 오류:', error);
                        $('#stock_chart').html('<div class="alert alert-warning">차트 데이터 형식 오류</div>');
                    }
                } else {
                    $('#stock_chart').html('<div class="alert alert-warning">차트를 표시할 수 없습니다.</div>');
                }
            })
            .fail(function(xhr, status, error) {
                console.error('차트 로드 실패:', error);
                $('#stock_chart').html('<div class="alert alert-danger">차트 로드 중 오류가 발생했습니다.</div>');
            });
        
        // 피처 데이터 요청
        $.get(`/api/stock_features/${paddedTicker}`)
            .done(function(data) {
                if (data.features) {
                    const tbody = $('#features_tbody');
                    tbody.empty();
                    for (const [feature, value] of Object.entries(data.features)) {
                        // JSON 문자열인지 확인하고 적절히 표시
                        let displayValue = value;
                        try {
                            const parsed = JSON.parse(value);
                            if (Array.isArray(parsed)) {
                                displayValue = parsed.join(', ');
                            } else if (typeof parsed === 'object') {
                                displayValue = JSON.stringify(parsed, null, 2);
                            }
                        } catch (e) {
                            // JSON이 아닌 경우 그대로 표시
                            displayValue = value;
                        }
                        tbody.append(`<tr><td>${feature}</td><td>${displayValue}</td></tr>`);
                    }
                } else {
                    $('#features_tbody').html('<tr><td colspan="2" class="text-center">피처 데이터를 찾을 수 없습니다.</td></tr>');
                }
            })
            .fail(function() {
                $('#features_tbody').html('<tr><td colspan="2" class="text-center text-danger">피처 데이터 로드 중 오류가 발생했습니다.</td></tr>');
            });
    }
    
    function updateAnalysisLog(message) {
        console.log('DEBUG: updateAnalysisLog 호출 -', message);
        const logContainer = $('#analysis_log');
        console.log('DEBUG: logContainer 찾음 -', logContainer.length);
        
        if (logContainer.length === 0) {
            console.error('DEBUG: analysis_log 요소를 찾을 수 없습니다!');
            return;
        }
        
        const currentLog = logContainer.text();
        logContainer.text(currentLog + message + '\n');
        logContainer.scrollTop(logContainer[0].scrollHeight);
        
        // 로그 통계 업데이트
        const lines = logContainer.text().split('\n').filter(line => line.trim());
        $('#total_logs').text(lines.length);
        $('#displayed_logs').text(lines.length);
        
        console.log('DEBUG: 로그 업데이트 완료 - 총', lines.length, '줄');
    }
    
    function handleAnalysisComplete(data) {
        // 분석 실행 중 플래그 해제
        window.analysisRunning = false;
        
        // 분석 시작 버튼 다시 활성화
        $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
        
        // 분석 중단 버튼 숨기기 (팝업 내부)
        $('#stop_analysis_btn_modal').hide();
        
        if (data.success) {
            $('#analysis_status').html('<span class="text-success">✅ 완료</span>');
            showToast('분석이 완료되었습니다.', 'success');
            setTimeout(function() {
                $('#analysis_modal').modal('hide');
                location.reload();
            }, 2000);
        } else {
            $('#analysis_status').html('<span class="text-danger">❌ 오류</span>');
            showToast('분석 중 오류가 발생했습니다: ' + data.error, 'danger');
            setTimeout(function() {
                $('#analysis_modal').modal('hide');
            }, 3000);
        }
    }
    
    function setupBacktestEvents() {
        // 백테스팅 폼 제출
        $('#backtest_form').on('submit', function(e) {
            e.preventDefault();
            startBacktest();
        });
        
        // 백테스팅 로그 수신
        if (socket) {
            socket.on('backtest_log', function(data) {
                updateBacktestLog(data.message);
            });
            
            socket.on('backtest_complete', function(data) {
                handleBacktestComplete(data);
            });
        }
    }
    
    function startBacktest() {
        // 모달 닫기
        $('#backtest_modal').modal('hide');
        
        // 진행 모달 표시
        $('#backtest_progress_modal').modal('show');
        
        // 폼 데이터 수집
        const formData = {
            capital: parseInt($('#capital').val()),
            max_hold: parseInt($('#max_hold').val()),
            take_profit: parseFloat($('#take_profit').val()),
            stop_loss: parseFloat($('#stop_loss').val()),
            top_n: parseInt($('#top_n').val()),
            buy_universe: parseInt($('#buy_universe').val()),
            transaction_fee: parseFloat($('#transaction_fee').val())
        };
        
        // 백테스팅 시작
        $.ajax({
            url: '/api/start_backtest',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                console.log('백테스팅 시작:', response);
                showToast('백테스팅이 시작되었습니다.', 'info');
            },
            error: function(xhr) {
                const error = JSON.parse(xhr.responseText);
                showToast('백테스팅 시작 중 오류: ' + error.error, 'danger');
                $('#backtest_progress_modal').modal('hide');
            }
        });
    }
    
    function updateBacktestLog(message) {
        const logContainer = $('#backtest_log');
        const currentLog = logContainer.text();
        logContainer.text(currentLog + message + '\n');
        logContainer.scrollTop(logContainer[0].scrollHeight);
    }
    
    function handleBacktestComplete(data) {
        if (data.success) {
            $('#backtest_progress_modal').modal('hide');
            showToast('백테스팅이 완료되었습니다.', 'success');
            // 페이지 새로고침하여 새로운 리포트 표시
            setTimeout(function() {
                location.reload();
            }, 1000);
        } else {
            showToast('백테스팅 중 오류가 발생했습니다: ' + data.error, 'danger');
            $('#backtest_progress_modal').modal('hide');
        }
    }
    
    function loadBacktestReport() {
        $.get('/static/backtest_report.html')
            .done(function(data) {
                $('#backtest_report').html(data);
            })
            .fail(function() {
                $('#backtest_report').html('<div class="alert alert-danger">리포트를 로드할 수 없습니다.</div>');
            });
    }
    
    function initializeFeatureChart() {
        // 피처 중요도 차트는 템플릿에서 직접 초기화됨
    }
    
    function initializeFeatureTable() {
        // DataTables 재초기화 방지
        if (!$.fn.DataTable.isDataTable('#feature_table')) {
            $('#feature_table').DataTable({
                pageLength: 25,
                order: [[1, 'desc']],
                language: {
                    "lengthMenu": "페이지당 _MENU_ 개씩 보기",
                    "zeroRecords": "데이터가 없습니다",
                    "info": "_START_ - _END_ / _TOTAL_ 개",
                    "infoEmpty": "0 개",
                    "infoFiltered": "(전체 _MAX_ 개 중 필터링됨)",
                    "search": "검색:",
                    "paginate": {
                        "first": "처음",
                        "last": "마지막",
                        "next": "다음",
                        "previous": "이전"
                    }
                }
            });
        }
    }
    
    function validateForm(form) {
        let isValid = true;
        
        form.find('input[required]').each(function() {
            if (!$(this).val()) {
                $(this).addClass('is-invalid');
                isValid = false;
            } else {
                $(this).removeClass('is-invalid');
            }
        });
        
        return isValid;
    }
    
    function formatNumberInput(input) {
        const value = parseFloat(input.val());
        if (!isNaN(value)) {
            if (input.attr('step') === '0.001') {
                input.val(value.toFixed(3));
            } else if (input.attr('step') === '0.1') {
                input.val(value.toFixed(1));
            } else {
                input.val(Math.round(value));
            }
        }
    }
    
    function showToast(message, type = 'info') {
        const toastId = 'toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        // 토스트 컨테이너가 없으면 생성
        if (!$('#toast-container').length) {
            $('body').append('<div id="toast-container" class="toast-container position-fixed top-0 end-0 p-3"></div>');
        }
        
        $('#toast-container').append(toastHtml);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: 5000
        });
        toast.show();
    }
    
    // 전역 함수로 노출
    window.showStockDetails = showStockDetails;
    window.showToast = showToast;
});
