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

        // 상태 폴링 (웹소켓 누락 대비)
        setInterval(syncServerStatus, 15000);
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
            const prevRunning = window.analysisRunning;
            if (serverRunning !== prevRunning) {
                console.log('서버-클라이언트 상태 불일치 감지, 동기화 중...');
                window.analysisRunning = serverRunning;
                updateAnalysisUI();
            }
            // 서버가 종료 상태인데 클라이언트가 실행 중 UI일 경우 리셋
            if (!serverRunning && prevRunning) {
                resetAnalysisState();
                $('#analysis_modal').modal('hide');
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

        // 가중치 수정 버튼 (주식추천/백테스트 공용)
        $(document).on('click', '.btn-edit-weights', function() {
            openWeightsModal();
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
                autoWidth: false,
                scrollX: true,
                scrollCollapse: false,
                fixedColumns: false,
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
                    { targets: [0], width: '70px', className: 'text-center' },      // 최종순위
                    { targets: [1], width: '90px', className: 'text-center' },      // 시장구분
                    { targets: [2], width: '140px' },                               // 종목명
                    { targets: [3], width: '90px', className: 'text-center' },      // 종목코드
                    { targets: [4], width: '110px', className: 'text-end' },        // 현재가
                    { targets: [5], width: '90px', className: 'text-end' },         // 등락율
                    { targets: [6], width: '110px', className: 'text-end' },        // 기준일가
                    { targets: [7], width: '90px', className: 'text-end' },         // 최종점수
                    { targets: [8], width: '110px', className: 'text-end' },        // RF상승확률
                    { targets: [9], width: '120px', className: 'text-end' },        // LGBM상승확률
                    { targets: [10], width: '110px', className: 'text-end' }        // 시가총액
                ],
                responsive: false,
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
            const basePrice = parseFloat(row.find('td:eq(6)').text().replace(/,/g, ''));
            
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
        // NASDAQ 티커는 그대로 사용
        const tickerStr = String(ticker).trim();
        
        // 차트 섹션 표시
        $('#chart_title').text(`📈 [${name}] 상세 차트`);
        $('#chart_section').show();
        
        // 피처 섹션 표시
        $('#features_title').text(`📊 ${name} (${ticker}) 분석 피처 데이터`);
        $('#features_section').show();
        
        // 차트 로딩
        $('#stock_chart').html('<div class="text-center"><i class="fas fa-spinner fa-spin fa-2x"></i><br>차트를 불러오는 중...</div>');
        
        // 차트 데이터 요청
        $.get(`/api/stock_chart/${encodeURIComponent(tickerStr)}`)
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
                        console.log('차트 렌더링 완료:', tickerStr);
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
        $.get(`/api/stock_features/${encodeURIComponent(tickerStr)}`)
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
        // 백테스팅 실행 버튼 (모달 내부)
        $('#execute_backtest_btn').on('click', function() {
            executeBacktest();
        });
        
        // 백테스팅 중단 버튼 (모달 내부)
        $('#stop_backtest_btn_modal').on('click', function() {
            stopBacktest();
        });
        
        // 모달이 열릴 때 서버 상태 확인 및 날짜 기본값 설정
        $('#backtest_modal').on('show.bs.modal', function() {
            // 날짜 필드 기본값 설정 (시작일: 1년 전, 종료일: 오늘)
            const today = new Date();
            const oneYearAgo = new Date(today);
            oneYearAgo.setFullYear(today.getFullYear() - 1);
            
            // 날짜를 YYYY-MM-DD 형식으로 변환
            const formatDate = function(date) {
                const year = date.getFullYear();
                const month = String(date.getMonth() + 1).padStart(2, '0');
                const day = String(date.getDate()).padStart(2, '0');
                return `${year}-${month}-${day}`;
            };
            
            // 날짜 필드가 비어있을 때만 기본값 설정
            if (!$('#modal_start_date').val()) {
                $('#modal_start_date').val(formatDate(oneYearAgo));
            }
            if (!$('#modal_end_date').val()) {
                $('#modal_end_date').val(formatDate(today));
            }
            
            checkBacktestServerStatus().then(function(serverRunning) {
                if (serverRunning) {
                    // 서버에서 백테스팅이 실행 중이면 실행 중 상태로 표시
                    showBacktestRunningState();
                    window.backtestRunning = true;
                } else {
                    // 서버에서 백테스팅이 실행 중이 아니면 준비 상태로 표시
                    showBacktestReadyState();
                    window.backtestRunning = false;
                }
            }).catch(function(error) {
                console.error('서버 상태 확인 실패:', error);
                // 서버 상태 확인 실패 시 준비 상태로 표시
                showBacktestReadyState();
            });
        });
        
        // 모달이 닫힐 때 백테스팅 종료 (서버 상태 확인 후)
        $('#backtest_modal').on('hidden.bs.modal', function() {
            if (window.backtestRunning) {
                // 서버 상태 확인 후 처리
                checkBacktestServerStatus().then(function(serverRunning) {
                    if (serverRunning && window.backtestRunning) {
                        // 서버와 클라이언트 모두 실행 중이면 중지 요청
                        stopBacktest();
                    } else {
                        // 상태 불일치 시 클라이언트 상태만 초기화
                        resetBacktestState();
                    }
                }).catch(function(error) {
                    console.error('서버 상태 확인 실패:', error);
                    // 서버 상태 확인 실패 시 안전하게 중지
                    stopBacktest();
                });
            }
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
    
    function executeBacktest() {
        // 날짜 입력값 가져오기
        const startDate = $('#modal_start_date').val();
        const endDate = $('#modal_end_date').val();
        
        // 날짜 유효성 검증
        if (!startDate || !endDate) {
            showToast('시작일과 종료일을 모두 입력해주세요.', 'warning');
            return;
        }
        
        const startDateObj = new Date(startDate);
        const endDateObj = new Date(endDate);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        endDateObj.setHours(0, 0, 0, 0);  // 시간 정보 제거하여 날짜만 비교
        
        // 종료일이 오늘보다 미래인지 확인 (오늘까지는 선택 가능)
        if (endDateObj > today) {
            showToast('종료일은 오늘 날짜를 초과할 수 없습니다.', 'warning');
            return;
        }
        
        // 시작일이 종료일보다 이후인지 확인
        if (startDateObj >= endDateObj) {
            showToast('시작일은 종료일보다 이전이어야 합니다.', 'warning');
            return;
        }
        
        // 폼 데이터 수집
        const formData = {
            capital: parseInt($('#modal_capital').val()),
            max_hold: parseInt($('#modal_max_hold').val()),
            take_profit: parseFloat($('#modal_take_profit').val()),
            stop_loss: parseFloat($('#modal_stop_loss').val()),
            top_n: parseInt($('#modal_top_n').val()),
            buy_universe: parseInt($('#modal_buy_universe').val()),
            transaction_fee: parseFloat($('#modal_transaction_fee').val()),
            start_date: startDate,
            end_date: endDate
        };
        
        // 서버 상태 확인 후 진행
        checkBacktestServerStatus().then(function(serverRunning) {
            if (serverRunning || window.backtestRunning) {
                showToast('이미 백테스팅이 실행 중입니다.', 'warning');
                return;
            }
            
            // 백테스팅 실행 중 상태로 변경
            showBacktestRunningState();
            
            // 즉시 플래그 설정 (중복 실행 방지)
            window.backtestRunning = true;
            
            // 백테스팅 시작 요청
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
                    // 오류 발생 시 상태 복구
                    resetBacktestState();
                    
                    const error = JSON.parse(xhr.responseText);
                    showToast('백테스팅 시작 중 오류: ' + error.error, 'danger');
                }
            });
        }).catch(function(error) {
            console.error('서버 상태 확인 실패:', error);
            showToast('서버 상태를 확인할 수 없습니다.', 'warning');
        });
    }
    
    function updateBacktestLog(message) {
        const logContainer = $('#backtest_log');
        const isProgressMessage = message.startsWith('[PROGRESS]');
        
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
    }
    
    function checkBacktestServerStatus() {
        return $.get('/api/backtest_status').then(function(data) {
            return data.backtest_running;
        }).catch(function() {
            return false; // 서버 오류 시 false로 가정
        });
    }
    
    function showBacktestReadyState() {
        // 백테스팅 준비 상태 표시
        $('#backtest_ready_section').show();
        $('#backtest_running_section').hide();
        
        // 백테스팅 실행 버튼 활성화
        $('#execute_backtest_btn').prop('disabled', false).html('<i class="fas fa-play me-1"></i>백테스팅 실행');
    }
    
    function showBacktestRunningState() {
        // 백테스팅 실행 중 상태 표시
        $('#backtest_ready_section').hide();
        $('#backtest_running_section').show();
        
        // 로그 초기화
        $('#backtest_log').text('');
    }
    
    function stopBacktest() {
        if (confirm('실행 중인 백테스팅을 중단하시겠습니까?')) {
            $.ajax({
                url: '/api/stop_backtest',
                method: 'POST',
                success: function(response) {
                    showToast('백테스팅이 중단되었습니다.', 'warning');
                    // 상태 복구
                    resetBacktestState();
                },
                error: function(xhr) {
                    const error = JSON.parse(xhr.responseText);
                    showToast('백테스팅 중단 중 오류: ' + error.error, 'danger');
                }
            });
        }
    }
    
    function resetBacktestState() {
        // 백테스팅 실행 중 플래그 해제
        window.backtestRunning = false;
        
        // 모달 상태를 준비 상태로 변경
        showBacktestReadyState();
    }
    
    function handleBacktestComplete(data) {
        if (data.success) {
            showToast('백테스팅이 완료되었습니다.', 'success');
            setTimeout(function() {
                $('#backtest_modal').modal('hide');
                location.reload();
            }, 2000);
        } else {
            showToast('백테스팅 중 오류가 발생했습니다: ' + data.error, 'danger');
            setTimeout(function() {
                $('#backtest_modal').modal('hide');
                // 오류 발생 시 상태 복구
                resetBacktestState();
            }, 3000);
        }
    }
    
    function loadBacktestReport() {
        const container = $('#backtest_report');
        // 로딩 아이콘 표시
        container.html(`
            <div class="text-center py-5">
                <i class="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
                <div class="h5 text-muted">백테스팅 리포트를 불러오는 중...</div>
                <div class="text-muted small">JSON 데이터를 분석하고 있습니다.</div>
            </div>
        `);
        
        // JSON 리포트 로드
        $.get('/api/backtest_report')
            .done(function(data) {
                renderBacktestReport(data);
            })
            .fail(function() {
                container.html('<div class="alert alert-danger">리포트를 로드할 수 없습니다.</div>');
            });
    }
    
    function renderBacktestReport(data) {
        const container = $('#backtest_report');
        container.empty();
        
        // 메타데이터 표시
        const metadata = data.metadata || {};
        const metrics = data.performance_metrics || {};
        const params = data.strategy_parameters || {};
        
        // 성과 지표 카드
        let metricsHtml = `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>성과 지표</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">초기 자본</div>
                                        <div class="h5 mb-0">${formatCurrency(metrics.initial_capital)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">최종 자산</div>
                                        <div class="h5 mb-0 ${metrics.final_asset >= metrics.initial_capital ? 'text-danger' : 'text-primary'}">${formatCurrency(metrics.final_asset)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">총수익률</div>
                                        <div class="h5 mb-0 ${metrics.total_return >= 0 ? 'text-danger' : 'text-primary'}">${formatPercent(metrics.total_return)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">연환산 수익률</div>
                                        <div class="h5 mb-0 ${metrics.annual_return >= 0 ? 'text-danger' : 'text-primary'}">${formatPercent(metrics.annual_return)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">샤프 지수</div>
                                        <div class="h5 mb-0">${metrics.sharpe_ratio.toFixed(2)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">최대 낙폭 (MDD)</div>
                                        <div class="h5 mb-0 text-primary">${formatPercent(metrics.mdd)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">승률</div>
                                        <div class="h5 mb-0">${formatPercent(metrics.win_rate)}</div>
                                    </div>
                                </div>
                                <div class="col-md-3 mb-3">
                                    <div class="text-center p-3 bg-light rounded">
                                        <div class="text-muted small mb-1">테스트 기간</div>
                                        <div class="h6 mb-0">${metadata.test_period ? metadata.test_period.start_date + ' ~ ' + metadata.test_period.end_date : 'N/A'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 전략 파라미터 카드
        let paramsHtml = `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0"><i class="fas fa-cogs me-2"></i>전략 파라미터</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3 mb-2"><strong>거래 수수료:</strong> ${params.transaction_fee_rate.toFixed(3)}%</div>
                                <div class="col-md-3 mb-2"><strong>증권거래세:</strong> ${params.securities_transaction_tax_rate.toFixed(2)}%</div>
                                <div class="col-md-3 mb-2"><strong>최대 보유 기간:</strong> ${params.max_hold_period}일</div>
                                <div class="col-md-3 mb-2"><strong>익절 목표:</strong> ${params.take_profit_pct.toFixed(2)}%</div>
                                <div class="col-md-3 mb-2"><strong>손절 라인:</strong> ${params.stop_loss_pct.toFixed(2)}%</div>
                                <div class="col-md-3 mb-2"><strong>매수 종목 수:</strong> ${params.top_n}개</div>
                                <div class="col-md-3 mb-2"><strong>매수 대상 범위:</strong> 상위 ${params.buy_universe_rank}위</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 차트 영역
        let chartHtml = `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5 class="mb-0"><i class="fas fa-chart-area me-2"></i>포트폴리오 성과 차트</h5>
                        </div>
                        <div class="card-body">
                            <div id="backtest_chart" style="height: 500px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 거래 로그 테이블
        let tradeLogHtml = '';
        if (data.trade_log && data.trade_log.length > 0) {
            tradeLogHtml = `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <h5 class="mb-0"><i class="fas fa-list me-2"></i>상세 매매 기록</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table id="backtest_trade_table" class="table table-striped table-hover">
                                        <thead class="table-dark">
                                            <tr>
                                                <th>거래일</th>
                                                <th>구분</th>
                                                <th>종목명</th>
                                                <th>종목코드</th>
                                                <th>시장구분</th>
                                                <th>매수일</th>
                                                <th>매도일</th>
                                                <th>보유기간</th>
                                                <th>매수가</th>
                                                <th>매도가</th>
                                                <th>수익률</th>
                                                <th>실현손익</th>
                                                <th>누적 실현손익</th>
                                                <th>매수금액</th>
                                                <th>시가총액(억달러)</th>
                                                <th>총자산</th>
                                                <th>최종점수</th>
                                                <th>RF상승확률</th>
                                                <th>LGBM상승확률</th>
                                            </tr>
                                        </thead>
                                        <tbody>
            `;
            
            // 시가총액 포맷팅 함수 (억달러 단위, 소수점 없음 / 값은 숫자만)
            function formatMarketCap(value) {
                if (value === null || value === undefined || isNaN(value)) return 'N/A';
                const v = value / 100000000; // 억달러(USD 100,000,000) 단위
                return new Intl.NumberFormat('en-US', {
                    maximumFractionDigits: 0
                }).format(v);
            }
            
            data.trade_log.forEach(function(trade) {
                const returnClass = trade.return !== null && trade.return !== undefined ? (trade.return >= 0 ? 'text-danger' : 'text-primary') : '';
                const profitClass = trade.profit !== null && trade.profit !== undefined ? (trade.profit >= 0 ? 'text-danger' : 'text-primary') : '';
                const cumulativeProfitClass = trade.cumulative_profit !== null && trade.cumulative_profit !== undefined ? (trade.cumulative_profit >= 0 ? 'text-danger' : 'text-primary') : '';
                
                tradeLogHtml += `
                    <tr>
                        <td>${trade.trade_date || 'N/A'}</td>
                        <td><span class="badge ${trade.type === 'buy' ? 'bg-primary' : 'bg-success'}">${trade.type === 'buy' ? '매수' : '매도'}</span></td>
                        <td>${trade.stock_name || 'N/A'}</td>
                        <td>${trade.ticker || 'N/A'}</td>
                        <td>${trade.market || 'N/A'}</td>
                        <td>${trade.buy_date || 'N/A'}</td>
                        <td>${trade.sell_date || 'N/A'}</td>
                        <td class="text-center">${trade.holding_period !== null && trade.holding_period !== undefined ? trade.holding_period + '일' : 'N/A'}</td>
                        <td class="text-end">${trade.buy_price !== null && trade.buy_price !== undefined ? formatCurrency(trade.buy_price) : 'N/A'}</td>
                        <td class="text-end">${trade.sell_price !== null && trade.sell_price !== undefined ? formatCurrency(trade.sell_price) : 'N/A'}</td>
                        <td class="text-end ${returnClass} fw-bold">${trade.return !== null && trade.return !== undefined ? formatPercent(trade.return) : 'N/A'}</td>
                        <td class="text-end ${profitClass} fw-bold">${trade.profit !== null && trade.profit !== undefined ? formatCurrency(trade.profit) : 'N/A'}</td>
                        <td class="text-end ${cumulativeProfitClass} fw-bold">${trade.cumulative_profit !== null && trade.cumulative_profit !== undefined ? formatCurrency(trade.cumulative_profit) : 'N/A'}</td>
                        <td class="text-end">${trade.buy_amount !== null && trade.buy_amount !== undefined ? formatCurrency(trade.buy_amount) : 'N/A'}</td>
                        <td class="text-end">${trade.buy_market_cap !== null && trade.buy_market_cap !== undefined ? formatMarketCap(trade.buy_market_cap) : 'N/A'}</td>
                        <td class="text-end">${trade.total_asset !== null && trade.total_asset !== undefined ? formatCurrency(trade.total_asset) : 'N/A'}</td>
                        <td class="text-end">${trade.final_score !== null && trade.final_score !== undefined ? trade.final_score.toFixed(2) : 'N/A'}</td>
                        <td class="text-end">${trade.ml_pred_proba !== null && trade.ml_pred_proba !== undefined ? (trade.ml_pred_proba * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td class="text-end">${trade.lgbm_pred_proba !== null && trade.lgbm_pred_proba !== undefined ? (trade.lgbm_pred_proba * 100).toFixed(2) + '%' : '-'}</td>
                    </tr>
                `;
            });
            
            tradeLogHtml += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        container.html(metricsHtml + paramsHtml + chartHtml + tradeLogHtml);
        
        // 차트 렌더링
        if (data.portfolio_history && data.portfolio_history.dates && data.portfolio_history.values) {
            renderBacktestChart(data);
        }
        
        // DataTables 초기화
        if ($('#backtest_trade_table').length) {
            if (!$.fn.DataTable.isDataTable('#backtest_trade_table')) {
                $('#backtest_trade_table').DataTable({
                    pageLength: 25,
                    order: [[0, 'desc']], // 최신 거래일 순
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
                    scrollX: true,
                    scrollCollapse: true,
                    responsive: false,  // scrollX와 함께 사용 시 responsive는 false
                    autoWidth: false,
                    columnDefs: [
                        { width: "100px", targets: 0 },  // 거래일
                        { width: "80px", targets: 1 },   // 구분
                        { width: "120px", targets: 2 }, // 종목명
                        { width: "100px", targets: 3 },  // 종목코드
                        { width: "90px", targets: 4 },   // 시장구분
                        { width: "100px", targets: 5 },  // 매수일
                        { width: "100px", targets: 6 },  // 매도일
                        { width: "80px", targets: 7 },   // 보유기간
                        { width: "100px", targets: 8 },  // 매수가
                        { width: "100px", targets: 9 },  // 매도가
                        { width: "100px", targets: 10 },  // 수익률
                        { width: "120px", targets: 11 }, // 실현손익
                        { width: "120px", targets: 12 }, // 누적 실현손익
                        { width: "120px", targets: 13 }, // 매수금액
                        { width: "120px", targets: 14 }, // 시가총액
                        { width: "120px", targets: 15 }, // 총자산
                        { width: "100px", targets: 16 }, // 최종점수
                        { width: "100px", targets: 17 }, // RF상승확률
                        { width: "100px", targets: 18 }  // LGBM상승확률
                    ]
                });
            }
        }
    }
    
    function renderBacktestChart(data) {
        const portfolioDates = data.portfolio_history.dates || [];
        const portfolioValues = data.portfolio_history.values || [];
        const ixicDates = (data.ixic_history && data.ixic_history.dates) ? data.ixic_history.dates : [];
        const ixicValues = (data.ixic_history && data.ixic_history.values) ? data.ixic_history.values : [];
        const tradeLog = data.trade_log || [];
        
        // 누적 실현손익 계산 (날짜별)
        const cumulativeProfitByDate = {};
        let cumulativeProfit = 0;
        
        // 거래 로그를 날짜순으로 정렬
        const sortedTrades = tradeLog.slice().sort(function(a, b) {
            const dateA = a.trade_date ? new Date(a.trade_date) : new Date(0);
            const dateB = b.trade_date ? new Date(b.trade_date) : new Date(0);
            return dateA - dateB;
        });
        
        // 매도 거래만 누적 실현손익 계산
        sortedTrades.forEach(function(trade) {
            if (trade.type === 'sell' && trade.profit !== null && trade.profit !== undefined) {
                cumulativeProfit += trade.profit;
                if (trade.trade_date) {
                    cumulativeProfitByDate[trade.trade_date] = cumulativeProfit;
                }
            }
        });
        
        // 포트폴리오 날짜에 대해 누적 실현손익 배열 생성
        const cumulativeProfitValues = [];
        let lastCumulativeProfit = 0;
        
        portfolioDates.forEach(function(date) {
            if (cumulativeProfitByDate[date] !== undefined) {
                lastCumulativeProfit = cumulativeProfitByDate[date];
            }
            cumulativeProfitValues.push(lastCumulativeProfit);
        });
        
        const traces = [];

        // 누적 실현손익 추적선 (USD, 축약 없이)
        if (portfolioDates.length > 0 && cumulativeProfitValues.length > 0) {
            traces.push({
                x: portfolioDates,
                y: cumulativeProfitValues,
                name: '누적 실현손익',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'royalblue', width: 2 }
            });
        }

        // IXIC 추적선 (초기 자본 기준 정규화된 값, USD)
        if (ixicDates.length > 0 && ixicValues.length > 0 && data.performance_metrics && data.performance_metrics.initial_capital) {
            const initialCapital = data.performance_metrics.initial_capital;
            const ixicDelta = ixicValues.map(function(val) {
                return (val - initialCapital);
            });

            traces.push({
                x: ixicDates,
                y: ixicDelta,
                name: 'IXIC (초기자본 대비)',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'grey', width: 1, dash: 'dash' }
            });
        }
        
        // 날짜 포맷팅 함수 (월/일 형식)
        function formatDate(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            if (isNaN(date.getTime())) return dateStr; // 유효하지 않은 날짜면 원본 반환
            const month = date.getMonth() + 1; // 0-11이므로 +1
            const day = date.getDate();
            return `${month}월 ${day}일`;
        }
        
        // X축 틱 값과 레이블 생성 (월/일 형식)
        const tickValues = [];
        const tickLabels = [];
        if (portfolioDates.length > 0) {
            const dateCount = portfolioDates.length;
            const tickInterval = Math.max(1, Math.floor(dateCount / 10)); // 최대 10개 틱
            
            for (let i = 0; i < dateCount; i += tickInterval) {
                const dateStr = portfolioDates[i];
                if (dateStr) {
                    const dateObj = new Date(dateStr);
                    if (!isNaN(dateObj.getTime())) {
                        tickValues.push(dateObj);
                        tickLabels.push(formatDate(dateStr));
                    }
                }
            }
            // 마지막 날짜도 포함
            if (dateCount > 0) {
                const lastDateStr = portfolioDates[dateCount - 1];
                const lastDateObj = new Date(lastDateStr);
                if (!isNaN(lastDateObj.getTime())) {
                    const lastTickValue = tickValues[tickValues.length - 1];
                    if (!lastTickValue || lastDateObj.getTime() !== lastTickValue.getTime()) {
                        tickValues.push(lastDateObj);
                        tickLabels.push(formatDate(lastDateStr));
                    }
                }
            }
        }
        
        // 호버 템플릿에 날짜/금액 포맷 적용 (USD)
        traces.forEach(function(trace) {
            if (trace.x && trace.x.length > 0) {
                trace.hovertemplate = '<b>%{customdata}</b><br>' +
                    trace.name + ': $%{y:,.2f}<br>' +
                    '<extra></extra>';
                trace.customdata = trace.x.map(formatDate);
            }
        });
        
        const layout = {
            title: {
                text: '<b>포트폴리오 성과 비교 (누적 실현손익)</b>',
                font: { size: 16 }
            },
            xaxis: {
                title: '날짜',
                type: 'date',
                tickmode: tickValues.length > 0 ? 'array' : 'auto',
                tickvals: tickValues.length > 0 ? tickValues : undefined,
                ticktext: tickLabels.length > 0 ? tickLabels : undefined,
                tickformat: tickValues.length === 0 ? '%m/%d' : undefined, // fallback 형식
                tickangle: -45
            },
            yaxis: {
                title: '누적 실현손익 (USD)',
                tickprefix: '$',
                tickformat: ',.0f'
            },
            hovermode: 'x unified',
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
            },
            margin: { l: 100, r: 50, t: 60, b: 80 }  // 아래 여백 증가 (날짜 표시 공간)
        };
        
        Plotly.newPlot('backtest_chart', traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    function formatCurrency(value) {
        // USD 풀표기 (축약 없음)
        if (value === null || value === undefined || isNaN(value)) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
    
    function formatPercent(value) {
        if (value === null || value === undefined || isNaN(value)) return 'N/A';
        const sign = value >= 0 ? '+' : '';
        return sign + (value * 100).toFixed(2) + '%';
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
    window.loadBacktestReport = loadBacktestReport;

    // ==============================
    // 가중치 수정 모달 (공용)
    // ==============================

    let weightsModalInstance = null;

    function openWeightsModal() {
        const modalEl = document.getElementById('weights_modal');
        if (!modalEl) {
            showToast('가중치 모달을 찾을 수 없습니다.', 'warning');
            return;
        }

        if (!weightsModalInstance) {
            weightsModalInstance = new bootstrap.Modal(modalEl);
        }

        // UI 초기화
        hideWeightsAlert();
        $('#weights_tbody').empty();
        $('#weights_sum').text('0');
        $('#weights_file_path').text('-');
        $('#weights_file_exists_badge').removeClass('bg-success bg-danger bg-secondary').addClass('bg-secondary').text('확인중');

        // 로드 후 표시
        loadWeightsIntoModal().then(function() {
            weightsModalInstance.show();
        }).catch(function(err) {
            showWeightsAlert(err?.message || '가중치 로딩 실패');
            weightsModalInstance.show();
        });
    }

    function showWeightsAlert(message) {
        $('#weights_modal_alert').removeClass('d-none').text(message);
    }

    function hideWeightsAlert() {
        $('#weights_modal_alert').addClass('d-none').text('');
    }

    function computeWeightsSum() {
        let sum = 0;
        $('#weights_tbody tr').each(function() {
            const val = parseFloat($(this).find('.weight-value').val());
            if (!isNaN(val)) sum += val;
        });
        $('#weights_sum').text(sum.toFixed(6));
    }

    function addWeightsRow(key = '', value = 0) {
        const displayNameMap = {
            'ml_pred_proba': 'ml_pred_proba (RF 상승확률)',
            'lgbm_pred_proba': 'lgbm_pred_proba (LGBM 상승확률)'
        };
        const displayName = displayNameMap[key] || key;
        const rowHtml = `
            <tr data-key="${escapeHtml(key)}">
                <td>
                    <div class="fw-semibold">${escapeHtml(displayName)}</div>
                    <div class="text-muted small">${escapeHtml(key)}</div>
                </td>
                <td>
                    <input type="range" class="form-range weight-slider mb-1" min="0" max="1" step="0.01" value="${value}">
                    <div class="input-group input-group-sm">
                        <input type="number" class="form-control form-control-sm weight-value text-end" step="0.000001" min="0" value="${value}">
                        <span class="input-group-text">%</span>
                    </div>
                </td>
            </tr>
        `;
        $('#weights_tbody').append(rowHtml);
        computeWeightsSum();
    }

    $(document).on('input', '.weight-slider', function() {
        const val = $(this).val();
        $(this).closest('td').find('.weight-value').val(val);
        computeWeightsSum();
    });

    $(document).on('input', '.weight-value', function() {
        const val = $(this).val();
        $(this).closest('td').find('.weight-slider').val(val);
        computeWeightsSum();
    });

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function loadWeightsIntoModal() {
        return $.get('/api/weights').then(function(data) {
            if (!data || !data.weights) throw new Error('가중치 응답이 올바르지 않습니다.');

            $('#weights_file_path').text(data.file_path || '-');
            if (data.file_exists) {
                $('#weights_file_exists_badge').removeClass('bg-secondary bg-danger').addClass('bg-success').text('존재');
            } else {
                $('#weights_file_exists_badge').removeClass('bg-secondary bg-success').addClass('bg-danger').text('없음');
            }

            const weights = data.weights || {};
            const allowedKeys = (data.allowed_keys || Object.keys(weights)).slice();
            // 서버에서 준 allowed_keys 순서를 우선 사용
            const keys = allowedKeys.length ? allowedKeys : Object.keys(weights).sort();
            $('#weights_tbody').empty();
            keys.forEach(function(k) {
                addWeightsRow(k, weights[k]);
            });
        });
    }

    function collectWeightsFromModal() {
        const weights = {};
        $('#weights_tbody tr').each(function() {
            const k = String($(this).attr('data-key') || '').trim();
            const vRaw = $(this).find('.weight-value').val();
            const v = parseFloat(vRaw);
            if (!k) return;
            weights[k] = isNaN(v) ? 0 : v;
        });
        return weights;
    }

    $(document).on('input', '.weight-value', function() {
        computeWeightsSum();
    });

    $(document).on('click', '#weights_save_btn', function() {
        hideWeightsAlert();

        const weights = collectWeightsFromModal();
        const normalize = $('#weights_normalize_toggle').is(':checked');

        if (Object.keys(weights).length === 0) {
            showWeightsAlert('저장할 가중치가 없습니다. (키를 입력하세요)');
            return;
        }

        // 저장 버튼 비활성화
        $('#weights_save_btn').prop('disabled', true).text('저장 중...');

        $.ajax({
            url: '/api/weights',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ weights: weights, normalize: normalize }),
            success: function(resp) {
                if (resp && resp.success) {
                    showToast('가중치가 저장되었습니다. (다음 분석/백테스트부터 반영)', 'success');
                    // 다시 로드해서 정규화 결과 반영
                    loadWeightsIntoModal().catch(function() {});
                } else {
                    showWeightsAlert(resp?.error || '저장 실패');
                }
            },
            error: function(xhr) {
                let msg = '저장 실패';
                try {
                    const j = JSON.parse(xhr.responseText);
                    msg = j.error || j.message || msg;
                } catch (e) {}
                showWeightsAlert(msg);
            },
            complete: function() {
                $('#weights_save_btn').prop('disabled', false).html('<i class="fas fa-save me-1"></i>저장');
            }
        });
    });

    // 전역으로도 노출 (디버그/재사용)
    window.openWeightsModal = openWeightsModal;
});
