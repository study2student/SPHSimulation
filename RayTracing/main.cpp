#include "DxLib.h"
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <omp.h>
#include <array>
#include <deque>
#undef min
#undef max

// 定数
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int PARTICLE_COLOR = GetColor(64, 164, 255);


// スレッド使用状況を視覚化するためのクラス
class ThreadVisualizer {
private:
    struct ThreadInfo {
        int threadId;
        bool isActive;
        int64_t lastActiveTime;
        std::string currentTask;
    };

    std::vector<ThreadInfo> threadInfos;
    int maxThreads;
    std::mutex infoMutex;
    bool isEnabled;

    // 追加する変数
	std::atomic<int> activeThreadsCount;// スレッド数をアトミックに管理

public:
    ThreadVisualizer() : isEnabled(true), activeThreadsCount(1) {
        maxThreads = omp_get_max_threads();
        threadInfos.resize(maxThreads);

        for (int i = 0; i < maxThreads; i++) {
            threadInfos[i].threadId = i;
            threadInfos[i].isActive = false;
            threadInfos[i].lastActiveTime = GetTickCount64();
            threadInfos[i].currentTask = "Idle";
        }
    }

    // 並列領域内からスレッド数を更新するメソッド
    void updateThreadCount() {
        if (!isEnabled) return;

        // 並列領域内でのみ正確なスレッド数を取得できる
#pragma omp single
        {
            activeThreadsCount = omp_get_num_threads();
        }

    }

    void setActive(const std::string& taskName) {
        if (!isEnabled) return;

        int threadId = omp_get_thread_num();

        std::lock_guard<std::mutex> lock(infoMutex);
        if (threadId >= 0 && threadId < maxThreads) {
            threadInfos[threadId].isActive = true;
            threadInfos[threadId].lastActiveTime = GetTickCount64();
            threadInfos[threadId].currentTask = taskName;
        }
    }

    void setInactive() {
        if (!isEnabled) return;

        int threadId = omp_get_thread_num();

        std::lock_guard<std::mutex> lock(infoMutex);
        if (threadId >= 0 && threadId < maxThreads) {
            threadInfos[threadId].isActive = false;
        }
    }

    void draw(int x, int y) {
        if (!isEnabled) return;

        int avtiveThreads = 0;
        // activeThreadsCountから取得したスレッド数を使う
		int currentThreads = activeThreadsCount.load();

        std::lock_guard<std::mutex> lock(infoMutex);

        // スレッド数情報の表示
        DrawFormatString(x, y, GetColor(255, 255, 255), "Active Threads: %d / %d", currentThreads, maxThreads);

        // 各スレッドの状態を可視化
        int boxSize = 15;
        int margin = 2;
        int64_t currentTime = GetTickCount64();

        for (int i = 0; i < maxThreads; i++) {
            int drawX = x + i * (boxSize + margin);
            int drawY = y + 20;

            // アクティブかどうかで色を変更
            int color;

            if (threadInfos[i].isActive) {
                color = GetColor(0, 255, 0); // 緑
                avtiveThreads++;
            }
            else {
                // 最近アクティブだったスレッド：黄色、それ以外：赤
                int64_t timeSinceActive = currentTime - threadInfos[i].lastActiveTime;
                if (timeSinceActive < 1000) {
                    color = GetColor(255, 255, 0); // 黄色
                }
                else {
                    color = GetColor(255, 0, 0); // 赤
                }
            }

            // スレッド状態を表す四角形を描画
            DrawBox(drawX, drawY, drawX + boxSize, drawY + boxSize, color, TRUE);

            // スレッドIDを表示
            DrawFormatString(drawX + 3, drawY + 2, GetColor(0, 0, 0), "%d", i);
        }

        // アクティブスレッド数を表示
        DrawFormatString(x, y + 40, GetColor(255, 255, 255), "Active: %d", avtiveThreads);

        // タスク情報(最初の3つのスレッドだけ表示)
        for (int i = 0; i < std::min(3, maxThreads); i++) {
            std::string taskInfo = "Thread " + std::to_string(i) + ": " + threadInfos[i].currentTask;
            if (taskInfo.length() > 30) {
                taskInfo = taskInfo.substr(0, 27) + "...";
            }
            DrawFormatString(x, y + 60 + i * 20, GetColor(200, 200, 200), "%s", taskInfo.c_str());
        }
    }

    void enable(bool enable) {
        isEnabled = enable;
    }

};

// グローバルインスタンス
ThreadVisualizer g_ThreadVisualizer; // 定義を追加

// 2Dベクトルクラス
struct Vec2 {
    float x, y;

    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(float _x, float _y) : x(_x), y(_y) {}

    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    Vec2& operator+=(const Vec2& other) { x += other.x; y += other.y; return *this; }
    Vec2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }

    float length() const { return std::sqrt(x * x + y * y); }
    float squaredLength() const { return x * x + y * y; }

    Vec2 normalized() const {
        float len = length();
        if (len < 1e-6f) return Vec2(0, 0);
        return Vec2(x / len, y / len);
    }

    float dot(const Vec2& other) const { return x * other.x + y * other.y; }
};

// 空間ハッシュの最適化版（整数ベースのキー）
class SpatialHash {
private:
    float cellSize;
    struct CellPos {
        int x, y;

        bool operator==(const CellPos& other) const {
            return x == other.x && y == other.y;
        }
    };

    struct CellPosHash {
        std::size_t operator()(const CellPos& pos) const {
            return std::hash<int>()(pos.x) ^ (std::hash<int>()(pos.y) << 1);
        }
    };

    std::unordered_map<CellPos, std::vector<int>, CellPosHash> grid;

public:
    SpatialHash(float size) : cellSize(size) {}

    void clear() {
        grid.clear();
    }

    void insert(int particleId, const Vec2& position) {
        int x = static_cast<int>(std::floor(position.x / cellSize));
        int y = static_cast<int>(std::floor(position.y / cellSize));
        CellPos cellPos{ x, y };
        grid[cellPos].push_back(particleId);
    }

    void reserve(int particleCount) {
        grid.reserve(particleCount / 3);
    }

    std::vector<int> getNeighbors(const Vec2& position, float radius) {
        std::vector<int> neighbors;
        neighbors.reserve(120); // より多くの近傍粒子を予め確保

        int x0 = static_cast<int>(std::floor((position.x - radius) / cellSize));
        int y0 = static_cast<int>(std::floor((position.y - radius) / cellSize));
        int x1 = static_cast<int>(std::floor((position.x + radius) / cellSize));
        int y1 = static_cast<int>(std::floor((position.y + radius) / cellSize));

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                CellPos cellPos{ x, y };
                auto it = grid.find(cellPos);
                if (it != grid.end()) {
                    const auto& cell = it->second;
                    neighbors.insert(neighbors.end(), cell.begin(), cell.end());
                }
            }
        }

        neighbors.shrink_to_fit(); // メモリの無駄を減らすために容量を縮小
        return neighbors;
    }
};

// 改良版SPH流体シミュレーションクラス
class ImprovedSPHFluidSimulator {

public:
    // 粒子データ
    struct Particle {
        Vec2 position;
        Vec2 velocity;
        Vec2 force;
        Vec2 oldPosition;    // 時間積分用の前ステップ位置
        Vec2 oldVelocity;    // 時間積分用の前ステップ速度
        Vec2 predictedPosition; // 予測位置
        float density;
        float pressure;
        float previousDensity; // 時間フィルタリング用
        float surfaceTension;  // 表面張力項
        float vorticityStrength; // 渦度
    };

private:
    // 改良されたパラメータ
    const float particleRadius = 0.05f;
    const float smoothingRadius = 0.12f;
    const float restDensity = 1000.0f;
    const float pressureCoeff = 250.0f;  // 圧力係数を調整
    const float viscosityCoeff = 0.15f;  // 粘性係数を調整
    const float surfaceTensionCoeff = 0.0725f; // 表面張力係数
    const float vorticityCoeff = 0.5f;   // 渦度係数
    const float gravityY = -9.8f;
    const float boundaryStiffness = 10000.0f;  // 境界の硬さ
    const float boundaryDampingCoeff = 256.0f; // 境界の減衰係数
    const float maxTimestep = 0.008f; // 最大タイムステップ
    const float minTimestep = 0.001f; // 最小タイムステップ
    const float courantFactor = 0.4f; // CFL条件のための係数
    const float massPerParticle = 1.0f;

    // 圧力補正のためのXSPHパラメータ
    const float xsphCoeff = 0.1f;

    // 時間積分のパラメータ
    const int substepCount = 2; // サブステップ数
    float currentTimestep = maxTimestep; // 適応的タイムステップ

    // 密度緩和パラメータ
    const float densityFilterCoeff = 0.1f;

    // カーネル関数の定数を事前計算
    const float poly6Coeff;
    const float spikyGradCoeff;
    const float viscosityLapCoeff;

    // 境界
    float minX = 0.0f, maxX = 10.0f;
    float minY = 0.0f, maxY = 10.0f;
    const float boundaryRadius = 0.1f; // 境界インタラクション半径

    std::vector<Particle> particles;
    SpatialHash spatialHash;

    // 粒子描画用のテクスチャ
    int particleTexture = -1;

    // パフォーマンス測定用
    std::deque<float> fpsHistory;
    std::deque<float> particleCountHistory;

    // 近傍情報のキャッシュ
    std::vector<std::vector<std::pair<int, float>>> neighborCache;

    // SPHカーネル関数
    float kernelPoly6(float r, float h) {
        if (r >= h) return 0.0f;
        float h2 = h * h;
        float r2 = r * r;
        return poly6Coeff * std::pow(h2 - r2, 3);
    }

    float kernelSpikyGradient(float r, float h) {
        if (r >= h) return 0.0f;
        return spikyGradCoeff * std::pow(h - r, 2);
    }

    float kernelViscosityLaplacian(float r, float h) {
        if (r >= h) return 0.0f;
        return viscosityLapCoeff * (h - r);
    }

    // 改良された境界力計算関数
    Vec2 calculateBoundaryForce(const Vec2& position, float radius) {
        Vec2 force(0, 0);

        // 左側境界との相互作用
        if (position.x < minX + radius) {
            float dist = position.x - minX;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.x += boundaryStiffness * penetration;
            }
        }

        // 右側境界との相互作用
        if (position.x > maxX - radius) {
            float dist = maxX - position.x;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.x -= boundaryStiffness * penetration;
            }
        }

        // 下側境界との相互作用
        if (position.y < minY + radius) {
            float dist = position.y - minY;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.y += boundaryStiffness * penetration;
            }
        }

        // 上側境界との相互作用
        if (position.y > maxY - radius) {
            float dist = maxY - position.y;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.y -= boundaryStiffness * penetration;
            }
        }

        return force;
    }

    // テクスチャの初期化
    void initParticleTexture() {
        // 円形の粒子テクスチャを作成
        int size = static_cast<int>(particleRadius * SCREEN_WIDTH / 5.0f) * 2;
        size = std::max(size, 8); // 最小サイズ

        particleTexture = MakeScreen(size, size, TRUE);

        SetDrawScreen(particleTexture);
        ClearDrawScreen();

        // ぼかし効果を持った粒子の描画
        int center = size / 2;
        float radius = size / 2.0f;

        // グラデーションで粒子を描画
        for (int r = 0; r <= radius; r++) {
            float alpha = 255.0f * (1.0f - r / radius);
            SetDrawBlendMode(DX_BLENDMODE_ALPHA, static_cast<int>(alpha));
            DrawCircle(center, center, radius - r, GetColor(255, 255, 255), TRUE);
        }

        SetDrawScreen(DX_SCREEN_BACK);
        SetDrawBlendMode(DX_BLENDMODE_NOBLEND, 255);
    }

public:
    ImprovedSPHFluidSimulator()
        : spatialHash(smoothingRadius),
        poly6Coeff(315.0f / (64.0f * 3.14159f * std::pow(smoothingRadius, 9))),
        spikyGradCoeff(-45.0f / (3.14159f * std::pow(smoothingRadius, 6))),
        viscosityLapCoeff(45.0f / (3.14159f * std::pow(smoothingRadius, 6))) {
        initParticleTexture();
    }

    ~ImprovedSPHFluidSimulator() {
        if (particleTexture != -1) {
            DeleteGraph(particleTexture);
        }
    }

    // 粒子の追加
    void addParticle(const Vec2& position, const Vec2& velocity = Vec2(0, 0)) {
        Particle particle;
        particle.position = position;
        particle.oldPosition = position;
        particle.velocity = velocity;
        particle.oldVelocity = velocity;
        particle.predictedPosition = position;
        particle.force = Vec2(0, 0);
        particle.density = restDensity;
        particle.previousDensity = restDensity;
        particle.pressure = 0.0f;
        particle.surfaceTension = 0.0f;
        particle.vorticityStrength = 0.0f;
        particles.push_back(particle);

        // キャッシュを更新
        neighborCache.resize(particles.size());
    }

    // 四角形領域に粒子を均等に配置
    void createParticleBlock(int rows, int cols, const Vec2& origin, float spacing) {
        // メモリを事前確保
        particles.reserve(particles.size() + rows * cols);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                // 少しランダム性を加える
                float jitterX = (rand() % 100 - 50) * 0.0002f;
                float jitterY = (rand() % 100 - 50) * 0.0002f;
                Vec2 pos(origin.x + x * spacing + jitterX, origin.y + y * spacing + jitterY);
                addParticle(pos);
            }
        }

        // 空間ハッシュのサイズを予約
        spatialHash.reserve(particles.size());
    }

    // 近傍粒子情報の計算とキャッシュ
    void computeNeighbors() {
        // 空間ハッシュの更新
        spatialHash.clear();
        for (size_t i = 0; i < particles.size(); i++) {
            spatialHash.insert(i, particles[i].position);
        }

        // 各粒子の近傍情報を計算
#pragma omp parallel
        {
            // スレッド数を更新(並列領域内で実行)
			g_ThreadVisualizer.updateThreadCount();

            #pragma omp for
            for (int i = 0; i < static_cast<int>(particles.size()); i++) {
                g_ThreadVisualizer.setActive("Computing neighbors");

                Particle& pi = particles[i];
                auto& neighbors = neighborCache[i];
                neighbors.clear();

                // 近傍粒子を取得
                auto neighborIndices = spatialHash.getNeighbors(pi.position, smoothingRadius);

                for (int j : neighborIndices) {
                    Particle& pj = particles[j];
                    Vec2 rij = pi.position - pj.position;
                    float r2 = rij.squaredLength();

                    if (r2 < smoothingRadius * smoothingRadius && i != j) {
                        float r = std::sqrt(r2);
                        neighbors.emplace_back(j, r);
                    }
                }
                g_ThreadVisualizer.setInactive();
            }
        }
    }
    // シミュレーションのメインステップ
    void update() {
        // 適応的タイムステップの計算
        updateTimestep();

        // サブステップで精度向上
        for (int substep = 0; substep < substepCount; substep++) {
            float substepTime = currentTimestep / substepCount;

            // ステップ1: 位置と速度の記録
            for (auto& p : particles) {
                p.oldPosition = p.position;
                p.oldVelocity = p.velocity;
            }

            // ステップ2: 予測位置の計算（explicit Euler）
            for (auto& p : particles) {
                p.predictedPosition = p.position + p.velocity * substepTime;
            }

            // ステップ3: 近傍情報の計算（予測位置ベース）
            computeNeighbors();

            // ステップ4: 密度と圧力の計算
            computeDensityPressure();

            // ステップ5: 表面張力と渦度の計算
            computeSurfaceTensionAndVorticity();

            // ステップ6: 力の計算
            computeForces();

            // ステップ7: 位置と速度の更新（Velocity Verlet法）
            integrateVerlet(substepTime);
        }
    }

    // 適応的タイムステップの計算
    void updateTimestep() {
        float maxVelocity = 0.0f;

        for (const auto& p : particles) {
            float vel = p.velocity.length();
            maxVelocity = std::max(maxVelocity, vel);
        }

        if (maxVelocity > 1e-6f) {
            // CFL条件に基づくタイムステップ計算
            float newTimestep = courantFactor * smoothingRadius / maxVelocity;

            // 急激な変化を避けるためにスムージング
            currentTimestep = 0.9f * currentTimestep + 0.1f * newTimestep;

            // 範囲制限
            currentTimestep = std::min(maxTimestep, std::max(minTimestep, currentTimestep));
        }
        else {
            currentTimestep = maxTimestep;
        }
    }

    // 密度と圧力の計算（改良版：密度フィルタリング）
    void computeDensityPressure() {
    #pragma omp parallel
        {
            g_ThreadVisualizer.updateThreadCount();

            #pragma omp for
            for (int i = 0; i < static_cast<int>(particles.size()); i++) {
                g_ThreadVisualizer.setActive("Computing desity");

                Particle& pi = particles[i];
                float newDensity = 0.0f;

                // 自分自身の寄与
                newDensity += massPerParticle * kernelPoly6(0, smoothingRadius);

                // 近傍粒子の寄与
                for (const auto& [j, r] : neighborCache[i]) {
                    if (i == j) continue;
                    newDensity += massPerParticle * kernelPoly6(r, smoothingRadius);
                }

                // 密度の時間フィルタリング
                if (pi.previousDensity > 0) {
                    pi.density = (1.0f - densityFilterCoeff) * pi.previousDensity + densityFilterCoeff * newDensity;
                }
                else {
                    pi.density = newDensity;
                }
                pi.previousDensity = pi.density;

                // 改良版圧力計算（Tait方程式）
                const float gamma = 7.0f;  // 気体定数に相当
                const float B = pressureCoeff * restDensity / gamma;

                float ratio = pi.density / restDensity;
                pi.pressure = B * (std::pow(ratio, gamma) - 1.0f);
                pi.pressure = std::max(0.0f, pi.pressure); // 負の圧力を防止

                g_ThreadVisualizer.setInactive();
            }
        }
    }

    // 表面張力と渦度の計算
    void computeSurfaceTensionAndVorticity() {
#pragma omp parallel
        {
            g_ThreadVisualizer.updateThreadCount();

            #pragma omp for
            for (int i = 0; i < static_cast<int>(particles.size()); i++) {
                g_ThreadVisualizer.setActive("Computing surface");

                Particle& pi = particles[i];

                // 表面法線の計算
                Vec2 normal(0, 0);
                float colorGradientSum = 0.0f;

                for (const auto& [j, r] : neighborCache[i]) {
                    if (i == j || r < 1e-6f) continue;

                    Particle& pj = particles[j];
                    Vec2 rij = pi.position - pj.position;
                    Vec2 gradW = rij * (1.0f / r) * kernelSpikyGradient(r, smoothingRadius);
                    normal += gradW * (massPerParticle / pj.density);
                    colorGradientSum += gradW.length();
                }

                // 表面張力の計算（表面曲率に比例）
                float surfaceStrength = normal.length();
                if (surfaceStrength > 0.1f) {  // 表面粒子の判定しきい値
                    // 表面張力の影響は表面粒子のみに適用
                    pi.surfaceTension = surfaceTensionCoeff * colorGradientSum;
                }
                else {
                    pi.surfaceTension = 0.0f;
                }

                // 渦度の計算
                Vec2 curl(0, 0);
                for (const auto& [j, r] : neighborCache[i]) {
                    if (i == j) continue;

                    Particle& pj = particles[j];
                    Vec2 rij = pi.position - pj.position;
                    Vec2 vij = pj.velocity - pi.velocity;

                    if (r > 1e-6f) {
                        Vec2 gradW = rij * (1.0f / r) * kernelSpikyGradient(r, smoothingRadius);
                        curl.x += (vij.y * gradW.x - vij.x * gradW.y);
                        curl.y += (vij.x * gradW.y - vij.y * gradW.x);
                    }
                }

                pi.vorticityStrength = vorticityCoeff * curl.length();

                g_ThreadVisualizer.setInactive();
            }
        }
    }

    // 力の計算（改良版：表面張力と渦度を追加）
    void computeForces() {
#pragma omp parallel
        {
            g_ThreadVisualizer.updateThreadCount();

            #pragma omp for
            for (int i = 0; i < static_cast<int>(particles.size()); i++) {
                g_ThreadVisualizer.setActive("Computing forces");

                Particle& pi = particles[i];
                // 重力
                pi.force = Vec2(0, gravityY * pi.density);

                // 境界力
                pi.force += calculateBoundaryForce(pi.position, boundaryRadius);

                // 粒子間力
                for (const auto& [j, r] : neighborCache[i]) {
                    if (i == j) continue;

                    Particle& pj = particles[j];
                    Vec2 rij = pi.position - pj.position;

                    // 非常に近い場合は単位ベクトルを調整
                    Vec2 rijNorm = (r > 1e-6f) ? rij * (1.0f / r) : Vec2(0, 1);

                    // 圧力項（改良版：両方の圧力を使用）
                    float pressureTerm = -massPerParticle * (pi.pressure / (pi.density * pi.density) +
                        pj.pressure / (pj.density * pj.density));
                    Vec2 pressureForce = rijNorm * pressureTerm * kernelSpikyGradient(r, smoothingRadius);
                    pi.force += pressureForce;

                    // 粘性項
                    Vec2 vij = pj.velocity - pi.velocity;
                    float viscosityTerm = viscosityCoeff * massPerParticle / pj.density;
                    Vec2 viscosityForce = vij * viscosityTerm * kernelViscosityLaplacian(r, smoothingRadius);
                    pi.force += viscosityForce;

                    // 表面張力項
                    if (pi.surfaceTension > 0) {
                        Vec2 surfaceForce = rijNorm * pi.surfaceTension * kernelPoly6(r, smoothingRadius);
                        pi.force += surfaceForce;
                    }

                    // 渦度項（渦を強化）
                    if (pi.vorticityStrength > 0) {
                        // 渦度に比例した力を加える（ランダムな方向）
                        float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f;
                        Vec2 vortexForce(std::cos(angle), std::sin(angle));
                        pi.force += vortexForce * pi.vorticityStrength * 0.01f;
                    }

                    // XSPH修正（速度の分散を低減）
                    float xsphWeight = xsphCoeff * massPerParticle * kernelPoly6(r, smoothingRadius) / pj.density;
                    pi.velocity += vij * xsphWeight;

                    g_ThreadVisualizer.setInactive();
                }
            }
        }
    }

    // Velocity Verlet法による積分（より高精度）
    void integrateVerlet(float dt) {
#pragma omp parallel
        {
            g_ThreadVisualizer.updateThreadCount();

            #pragma omp for
            for (int i = 0; i < static_cast<int>(particles.size()); i++) {
                g_ThreadVisualizer.setActive("Computing intergration");

                Particle& p = particles[i];

                // 加速度 = 力 / 質量
                Vec2 acceleration = p.force * (1.0f / p.density);

                // 速度の半ステップ更新
                Vec2 halfVelocity = p.velocity + acceleration * (dt * 0.5f);

                // 位置を更新
                p.position += halfVelocity * dt;

                // 境界処理
                processBoundaryConditions(p);

                // 最終的な速度更新
                p.velocity = halfVelocity + acceleration * (dt * 0.5f);

                g_ThreadVisualizer.setInactive();
            }
        }
    }

    // 改良された境界条件処理
    void processBoundaryConditions(Particle& p) {
        // 左側境界
        if (p.position.x < minX) {
            p.position.x = minX + (minX - p.position.x) * 0.5f;
            p.velocity.x = -p.velocity.x * std::exp(-boundaryDampingCoeff * /*fabs*/(p.velocity.x));
        }
        // 右側境界
        if (p.position.x > maxX) {
            p.position.x = maxX - (p.position.x - maxX) * 0.5f;
            p.velocity.x = -p.velocity.x * std::exp(-boundaryDampingCoeff * /*fabs*/(p.velocity.x));
        }
        // 下側境界
        if (p.position.y < minY) {
            p.position.y = minY + (minY - p.position.y) * 0.5f;
            p.velocity.y = -p.velocity.y * std::exp(-boundaryDampingCoeff * /*fabs*/(p.velocity.y));
        }
        // 上側境界
        if (p.position.y > maxY) {
            p.position.y = maxY - (p.position.y - maxY) * 0.5f;
            p.velocity.y = -p.velocity.y * std::exp(-boundaryDampingCoeff * /*fabs*/(p.velocity.y));
        }
    }

    // レンダリング用に粒子の位置を取得
    const std::vector<Particle>& getParticles() const {
        return particles;
    }

    // シミュレーション領域の設定
    void setBoundary(float _minX, float _maxX, float _minY, float _maxY) {
        minX = _minX;
        maxX = _maxX;
        minY = _minY;
        maxY = _maxY;
    }

    // シミュレーション領域を描画する
    void drawBoundary() {
        // シミュレーション座標をスクリーン座標に変換
        int x1 = static_cast<int>(minX * SCREEN_WIDTH / 10.0f);
        int y1 = SCREEN_HEIGHT - static_cast<int>(maxY * SCREEN_HEIGHT / 10.0f);
        int x2 = static_cast<int>(maxX * SCREEN_WIDTH / 10.0f);
        int y2 = SCREEN_HEIGHT - static_cast<int>(minY * SCREEN_HEIGHT / 10.0f);

        // 境界を白い四角で描画
        DrawBox(x1, y1, x2, y2, GetColor(255, 255, 255), FALSE);
    }

    // 粒子を描画する - テクスチャ使用の最適化版
    void drawParticles() {
        int textureSize = GetDrawFormatStringWidth("%d", particleTexture);
        float scaleFactor = particleRadius * SCREEN_WIDTH / 5.0f / (textureSize / 2);

        // カリング用の画面範囲
        int screenLeft = -textureSize;
        int screenTop = -textureSize;
        int screenRight = SCREEN_WIDTH + textureSize;
        int screenBottom = SCREEN_HEIGHT + textureSize;

        SetDrawBlendMode(DX_BLENDMODE_ALPHA, 255);

        for (const auto& p : particles) {
            // シミュレーション座標をスクリーン座標に変換
            int x = static_cast<int>(p.position.x * SCREEN_WIDTH / 10.0f);
            int y = SCREEN_HEIGHT - static_cast<int>(p.position.y * SCREEN_HEIGHT / 10.0f);

            // 画面外の粒子はスキップ
            if (x < screenLeft || x > screenRight || y < screenTop || y > screenBottom) {
                continue;
            }

            // 密度と表面張力に応じてカラーを調整
            float densityFactor = std::min(1.0f, p.density / (restDensity * 2.0f));
            float surfaceFactor = p.surfaceTension * 10.0f;

            int r = static_cast<int>(64 + 191 * densityFactor);
            int g = static_cast<int>(164 - 100 * densityFactor + 100 * surfaceFactor);
            int b = static_cast<int>(255 - 100 * densityFactor);
            int particleColor = GetColor(r, g, b);

            // テクスチャを描画
            DrawRotaGraph(x, y, scaleFactor, 0, particleTexture, TRUE);
        }

        SetDrawBlendMode(DX_BLENDMODE_NOBLEND, 255);
    }

    // パフォーマンス情報を描画
    void drawPerformanceInfo() {
        // 現在のFPSを計算し、履歴に追加
        float currentFPS = GetFPS();
        fpsHistory.push_back(currentFPS);
        if (fpsHistory.size() > 60) {
            fpsHistory.pop_front();
        }

        // 粒子数履歴の更新
        particleCountHistory.push_back(static_cast<float>(particles.size()));
        if (particleCountHistory.size() > 60) {
            particleCountHistory.pop_front();
        }

        // 平均FPSの計算
        float avgFPS = 0.0f;
        for (float fps : fpsHistory) {
            avgFPS += fps;
        }
        avgFPS /= fpsHistory.size();

        // 情報表示
        DrawFormatString(10, 10, GetColor(255, 255, 255), "FPS: %.1f", avgFPS);
        DrawFormatString(10, 30, GetColor(255, 255, 255), "Particles: %d", particles.size());
        DrawFormatString(10, 50, GetColor(255, 255, 255), "Timestep: %.4f", currentTimestep);

        // スレッド数使用状況の可視化
        g_ThreadVisualizer.draw(10, 90);
    }

    // マウス位置に粒子を追加
    void addParticlesAtMouse(int num) {
        int mouseX, mouseY;
        GetMousePoint(&mouseX, &mouseY);

        // マウス位置をシミュレーション座標に変換
        float simX = mouseX * 10.0f / SCREEN_WIDTH;
        float simY = (SCREEN_HEIGHT - mouseY) * 10.0f / SCREEN_HEIGHT;

        // マウス位置周辺にランダムに粒子を追加
        for (int i = 0; i < num; i++) {
            float offsetX = (rand() % 100 - 50) * 0.01f;
            float offsetY = (rand() % 100 - 50) * 0.01f;
            Vec2 pos(simX + offsetX, simY + offsetY);
            Vec2 vel(offsetX * 2.0f, offsetY * 2.0f);
            addParticle(pos, vel);
        }
    }

    // ダム決壊シナリオをセットアップ
    void setupDamBreakScenario() {
        // 既存の粒子をクリア
        particles.clear();
        neighborCache.clear();

        // 左側に水の塊を作成
        float spacing = particleRadius * 1.5f;
        int rows = static_cast<int>((maxY - minY) * 0.8f / spacing);
        int cols = static_cast<int>((maxX - minX) * 0.4f / spacing);

        Vec2 origin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, origin, spacing);
    }

    // 水滴落下シナリオをセットアップ
    void setupDropScenario() {
        // 既存の粒子をクリア
        particles.clear();
        neighborCache.clear();

        // 水の池を作成（下部に少量の水）
        float spacing = particleRadius * 1.5f;
        int rows = 3;
        int cols = static_cast<int>((maxX - minX) * 0.8f / spacing);

        Vec2 origin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, origin, spacing);

        // 上部に水滴を作成
        int dropRadius = 5;
        Vec2 dropCenter(maxX * 0.5f, maxY * 0.8f);

        for (int y = -dropRadius; y <= dropRadius; y++) {
            for (int x = -dropRadius; x <= dropRadius; x++) {
                float dist = std::sqrt(x * x + y * y);
                if (dist <= dropRadius) {
                    Vec2 pos(dropCenter.x + x * spacing, dropCenter.y + y * spacing);
                    addParticle(pos, Vec2(0, -5.0f)); // 下向きの初速度を与える
                }
            }
        }
    }

    // 二流体混合シナリオをセットアップ
    void setupTwoFluidScenario() {
        // 既存の粒子をクリア
        particles.clear();
        neighborCache.clear();

        // 左側と右側にそれぞれ異なる密度の流体を配置
        float spacing = particleRadius * 1.5f;
        int rows = static_cast<int>((maxY - minY) * 0.6f / spacing);
        int cols = static_cast<int>((maxX - minX) * 0.4f / spacing) / 2;

        // 左側の流体（高密度）
        Vec2 leftOrigin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, leftOrigin, spacing);

        // 右側の流体（低密度）
        Vec2 rightOrigin(maxX - cols * spacing, minY + spacing);
        int rightParticleStart = particles.size();
        createParticleBlock(rows, cols, rightOrigin, spacing);

        // 右側の粒子は異なる特性を持つように設定
        for (size_t i = rightParticleStart; i < particles.size(); i++) {
            particles[i].density = restDensity * 0.7f; // 低密度
            particles[i].previousDensity = restDensity * 0.7f;
        }
    }
};

// メイン関数
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // DXライブラリの初期化
    SetGraphMode(SCREEN_WIDTH, SCREEN_HEIGHT, 32);
    ChangeWindowMode(TRUE);
    SetWindowText("Improved SPH Fluid Simulation");
    if (DxLib_Init() == -1) return -1;
    SetDrawScreen(DX_SCREEN_BACK);

    // OpenMPの初期化
    int maxThreads = omp_get_max_threads();
	int numThreads = std::max(4, maxThreads); // 最小4スレッド
	omp_set_num_threads(numThreads);
    
    // スレッド情報を表示
	printf("Max threads available: %d, Using threads: %d\n", maxThreads, numThreads);

    // シミュレータの作成
    ImprovedSPHFluidSimulator simulator;
    simulator.setBoundary(0.5f, 9.5f, 0.5f, 9.5f);
    simulator.setupDamBreakScenario();

    // シナリオ選択用の変数
    int currentScenario = 0; // 0: ダム決壊, 1: 水滴, 2: 二流体

    int frameCount = 0;
    bool isRunning = true;
    bool isPaused = false;

    // メインループ
    while (isRunning && ProcessMessage() == 0) {
        ClearDrawScreen();

        // キー入力処理
        if (CheckHitKey(KEY_INPUT_ESCAPE)) isRunning = false;
        if (CheckHitKey(KEY_INPUT_SPACE) && frameCount % 15 == 0) isPaused = !isPaused;

        // マウス入力処理
        if ((GetMouseInput() & MOUSE_INPUT_LEFT) != 0 && frameCount % 5 == 0) {
            simulator.addParticlesAtMouse(5);
        }

        // シナリオ切り替え
        if (CheckHitKey(KEY_INPUT_1) && frameCount % 15 == 0) {
            currentScenario = 0;
            simulator.setupDamBreakScenario();
        }
        if (CheckHitKey(KEY_INPUT_2) && frameCount % 15 == 0) {
            currentScenario = 1;
            simulator.setupDropScenario();
        }
        if (CheckHitKey(KEY_INPUT_3) && frameCount % 15 == 0) {
            currentScenario = 2;
            simulator.setupTwoFluidScenario();
        }

        // シミュレーション更新
        if (!isPaused) {
            simulator.update();
        }

        // 描画
        simulator.drawBoundary();
        simulator.drawParticles();
        simulator.drawPerformanceInfo();

        // シナリオ情報表示
        const char* scenarioNames[] = { "Dam Break", "Droplet", "Two Fluids" };
        DrawFormatString(10, 70, GetColor(255, 255, 255), "Scenario: %s", scenarioNames[currentScenario]);
        DrawString(10, SCREEN_HEIGHT - 60, "Controls:", GetColor(200, 200, 200));
        DrawString(10, SCREEN_HEIGHT - 40, "1-3: Change Scenario, Space: Pause, Left Click: Add Particles", GetColor(200, 200, 200));
        DrawString(10, SCREEN_HEIGHT - 20, "Esc: Exit", GetColor(200, 200, 200));

        // 画面の更新
        ScreenFlip();
        frameCount++;
    }

    DxLib_End();
    return 0;
}