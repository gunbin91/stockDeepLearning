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
        
        // 서버 상태 동기화
        syncServerStatus();
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
            // 연결 시 서버 상태 동기화
            syncServerStatus();
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
    
    function syncServerStatus() {
        // 서버 상태 확인
        checkServerStatus().then(function(serverRunning) {
            if (serverRunning !== window.analysisRunning) {
                console.log('서버-클라이언트 상태 불일치 감지, 동기화 중...');
                window.analysisRunning = serverRunning;
                updateAnalysisUI();
            }
        }).catch(function(error) {
            console.error('서버 상태 확인 실패:', error);
        });
    }
    
    function checkServerStatus() {
        return $.get('/api/analysis_status').then(function(data) {
            return data.analysis_running;
        }).catch(function() {
            return false; // 서버 오류 시 false로 가정
        });
    }
    
    function updateAnalysisUI() {
        if (window.analysisRunning) {
            $('#start_analysis_btn').prop('disabled', true).text('분석 실행 중...');
            $('#stop_analysis_btn_modal').show();
        } else {
            $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
            $('#stop_analysis_btn_modal').hide();
        }
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
        
        
        // 분석 실행 버튼 (모달 내부)
        $('#execute_analysis_btn').on('click', function() {
            executeAnalysis();
        });
        
        // 분석 중단 버튼 (팝업 내부)
        $('#stop_analysis_btn_modal').on('click', function() {
            stopAnalysis();
        });
        
        // 모달이 열릴 때 서버 상태 확인
        $('#analysis_modal').on('show.bs.modal', function() {
            checkServerStatus().then(function(serverRunning) {
                if (serverRunning) {
                    // 서버에서 분석이 실행 중이면 실행 중 상태로 표시
                    showAnalysisRunningState();
                    window.analysisRunning = true;
                    $('#start_analysis_btn').prop('disabled', true).text('분석 실행 중...');
                } else {
                    // 서버에서 분석이 실행 중이 아니면 준비 상태로 표시
                    showAnalysisReadyState();
                    window.analysisRunning = false;
                    $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
                }
            }).catch(function(error) {
                console.error('서버 상태 확인 실패:', error);
                // 서버 상태 확인 실패 시 준비 상태로 표시
                showAnalysisReadyState();
            });
        });
        
        // 팝업이 닫힐 때 분석 종료 (서버 상태 확인 후)
        $('#analysis_modal').on('hidden.bs.modal', function() {
            if (window.analysisRunning) {
                // 서버 상태 확인 후 처리
                checkServerStatus().then(function(serverRunning) {
                    if (serverRunning && window.analysisRunning) {
                        // 서버와 클라이언트 모두 실행 중이면 중지 요청
                        stopAnalysis();
                    } else {
                        // 상태 불일치 시 클라이언트 상태만 초기화
                        resetAnalysisState();
                    }
                }).catch(function(error) {
                    console.error('서버 상태 확인 실패:', error);
                    // 서버 상태 확인 실패 시 안전하게 중지
                    stopAnalysis();
                });
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
                updateAnalysisLog(data.message);
            });
            
            socket.on('analysis_complete', function(data) {
                handleAnalysisComplete(data);
            });
        }
    }
    
    function startAnalysis() {
        // 메인 페이지의 분석 기준일을 모달로 복사
        const analysisDate = $('#analysis_date').val();
        $('#modal_analysis_date').val(analysisDate);
        
        // 모달 상태 초기화
        showAnalysisReadyState();
        
        // 모달 표시
        $('#analysis_modal').modal('show');
    }
    
    function resetAnalysisState() {
        // 분석 실행 중 플래그 해제
        window.analysisRunning = false;
        
        // 분석 시작 버튼 다시 활성화
        $('#start_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>새로운 분석 시작');
        
        // 모달 상태를 준비 상태로 변경
        showAnalysisReadyState();
    }
    
    function showAnalysisReadyState() {
        // 분석 준비 상태 표시
        $('#analysis_ready_section').show();
        $('#analysis_running_section').hide();
        
        // 분석 실행 버튼 활성화
        $('#execute_analysis_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>분석 실행');
    }
    
    function showAnalysisRunningState() {
        // 분석 실행 중 상태 표시
        $('#analysis_ready_section').hide();
        $('#analysis_running_section').show();
        
        // 로그 초기화
        $('#analysis_log').text('');
        $('#total_logs').text('0');
        $('#displayed_logs').text('0');
        $('#analysis_status').html('🔄 진행 중');
    }
    
    function executeAnalysis() {
        const analysisDate = $('#modal_analysis_date').val();
        if (!analysisDate) {
            showToast('분석 기준일을 선택해주세요.', 'warning');
            return;
        }
        
        // 서버 상태 확인 후 진행
        checkServerStatus().then(function(serverRunning) {
            if (serverRunning || window.analysisRunning) {
                showToast('이미 분석이 실행 중입니다.', 'warning');
                return;
            }
            
            // 분석 실행 중 상태로 변경
            showAnalysisRunningState();
            
            // 즉시 플래그 설정 및 버튼 비활성화 (중복 실행 방지)
            window.analysisRunning = true;
            $('#start_analysis_btn').prop('disabled', true).text('분석 실행 중...');
            
            // 분석 시작 요청
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
                    // 오류 발생 시 상태 복구
                    resetAnalysisState();
                    
                    const error = JSON.parse(xhr.responseText);
                    showToast('분석 시작 중 오류: ' + error.error, 'danger');
                    $('#analysis_modal').modal('hide');
                }
            });
        }).catch(function(error) {
            console.error('서버 상태 확인 실패:', error);
            showToast('서버 상태를 확인할 수 없습니다.', 'warning');
        });
    }
    
    
    function stopAnalysis() {
        if (confirm('실행 중인 분석을 중단하시겠습니까?')) {
            $.ajax({
                url: '/api/stop_analysis',
                method: 'POST',
                success: function(response) {
                    showToast('분석이 중단되었습니다.', 'warning');
                    // 상태 복구
                    resetAnalysisState();
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
        const logContainer = $('#analysis_log');
        
        if (logContainer.length === 0) {
            return;
        }
        
        // 진행률 패턴 감지 (터미널 스타일 처리)
        const progressPattern = /^\[PROGRESS\]/;
        const isProgressMessage = progressPattern.test(message);
        
        // 진행률 메시지가 아닌 경우에만 중복 필터링 적용
        if (!isProgressMessage) {
            // 중복 메시지 필터링 (마지막 3줄과 비교)
            const currentLog = logContainer.text();
            const lines = currentLog.split('\n').filter(line => line.trim());
            const lastThreeLines = lines.slice(-3);
            
            // 동일한 메시지가 최근 3줄에 있으면 무시
            if (lastThreeLines.includes(message)) {
                return;
            }
        }
        
        const currentLog = logContainer.text();
        
        if (isProgressMessage) {
            // 진행률 메시지는 같은 줄에서 업데이트 (터미널 스타일)
            const currentLogLines = currentLog.split('\n').filter(line => line.trim() !== '');
            const lastLine = currentLogLines.length > 0 ? currentLogLines[currentLogLines.length - 1] : '';
            const isLastLineProgress = lastLine.startsWith('[PROGRESS]');
            
            // 마지막 줄이 진행률이면 덮어쓰기, 아니면 새 줄 추가
            if (currentLogLines.length > 0 && isLastLineProgress) {
                // 덮어쓰기: 마지막 진행률 메시지를 새 메시지로 교체
                currentLogLines[currentLogLines.length - 1] = message;
                logContainer.text(currentLogLines.join('\n'));
            } else {
                // 새 줄에 추가
                logContainer.text(currentLog + message + '\n');
            }
        } else {
            // 일반 로그는 새 줄에 추가
            logContainer.text(currentLog + message + '\n');
        }
        
        logContainer.scrollTop(logContainer[0].scrollHeight);
        
        // 로그 통계 업데이트
        const finalLines = logContainer.text().split('\n').filter(line => line.trim());
        $('#total_logs').text(finalLines.length);
        $('#displayed_logs').text(finalLines.length);
    }
    
    function handleAnalysisComplete(data) {
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
                // 오류 발생 시 상태 복구
                resetAnalysisState();
            }, 3000);
        }
        // 분석 완료 시 서버 상태 동기화
        syncServerStatus();
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
