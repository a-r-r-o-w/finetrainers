@echo off
echo ===================================
echo finetrainers Docker セットアップ
echo ===================================
echo.

REM Dockerがインストールされているか確認
docker -v > nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Dockerがインストールされていないか、Docker Desktopが起動していません。
    echo 1. Docker Desktopをインストールしてください: https://www.docker.com/products/docker-desktop/
    echo 2. Docker Desktopが既にインストールされている場合は、起動してください。
    echo   （スタートメニューから「Docker Desktop」を検索して起動）
    exit /b 1
)

REM Docker Desktopが実際に起動しているか確認（追加チェック）
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Docker Desktopエンジンに接続できません。
    echo Docker Desktopが起動していることを確認してください。
    echo タスクトレイにDockerアイコンが表示されていない場合は、スタートメニューから起動してください。
    exit /b 1
)

REM Docker Composeがインストールされているか確認
docker-compose -v > nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Docker Composeがインストールされていません。
    echo Docker Desktopには通常Docker Composeが含まれています。
    exit /b 1
)

echo [情報] Dockerが正常にインストールされています。

REM フォルダ作成
if not exist "datasets" (
    echo [情報] datasetsフォルダを作成しています...
    mkdir datasets
)

if not exist "outputs" (
    echo [情報] outputsフォルダを作成しています...
    mkdir outputs
)

echo.
echo [情報] Dockerイメージをビルドしています...
docker-compose build

if %errorlevel% neq 0 (
    echo [エラー] Dockerイメージのビルドに失敗しました。
    echo 可能性のある原因:
    echo - Docker Desktopが正常に動作していない
    echo - WSL2の設定に問題がある
    echo - ディスク容量不足
    echo Docker Desktopが正しく設定されているか確認し、必要に応じてWindowsを再起動してください。
    exit /b 1
)

echo.
echo [情報] セットアップが完了しました！
echo.
echo 使用方法:
echo  1. Docker Desktopが実行中であることを確認
echo  2. docker-compose up -d        (コンテナを起動)
echo  3. docker exec -it finetrainers bash  (コンテナに接続)
echo  4. docker-compose down         (コンテナを停止)
echo.
echo 詳細は README-docker.md を参照してください。
echo.