#include "DxLib.h"
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

// �萔
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int PARTICLE_COLOR = GetColor(64, 164, 255);

// 2D�x�N�g���N���X
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

// ��ԃn�b�V���̍œK���Łi�����x�[�X�̃L�[�j
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
        neighbors.reserve(60); // ��葽���̋ߖT���q��\�ߊm��

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

        neighbors.shrink_to_fit(); // �������̖��ʂ����炷���߂ɗe�ʂ��k��
        return neighbors;
    }
};

// ���ǔ�SPH���̃V�~�����[�V�����N���X
class ImprovedSPHFluidSimulator {

public:
    // ���q�f�[�^
    struct Particle {
        Vec2 position;
        Vec2 velocity;
        Vec2 force;
        Vec2 oldPosition;    // ���Ԑϕ��p�̑O�X�e�b�v�ʒu
        Vec2 oldVelocity;    // ���Ԑϕ��p�̑O�X�e�b�v���x
        Vec2 predictedPosition; // �\���ʒu
        float density;
        float pressure;
        float previousDensity; // ���ԃt�B���^�����O�p
        float surfaceTension;  // �\�ʒ��͍�
        float vorticityStrength; // �Q�x
    };

private:
    // ���ǂ��ꂽ�p�����[�^
    const float particleRadius = 0.05f;
    const float smoothingRadius = 0.12f;
    const float restDensity = 1000.0f;
    const float pressureCoeff = 250.0f;  // ���͌W���𒲐�
    const float viscosityCoeff = 0.15f;  // �S���W���𒲐�
    const float surfaceTensionCoeff = 0.0725f; // �\�ʒ��͌W��
    const float vorticityCoeff = 0.5f;   // �Q�x�W��
    const float gravityY = -9.8f;
    const float boundaryStiffness = 10000.0f;  // ���E�̍d��
    const float boundaryDampingCoeff = 256.0f; // ���E�̌����W��
    const float maxTimestep = 0.008f; // �ő�^�C���X�e�b�v
    const float minTimestep = 0.001f; // �ŏ��^�C���X�e�b�v
    const float courantFactor = 0.4f; // CFL�����̂��߂̌W��
    const float massPerParticle = 1.0f;

    // ���͕␳�̂��߂�XSPH�p�����[�^
    const float xsphCoeff = 0.1f;

    // ���Ԑϕ��̃p�����[�^
    const int substepCount = 2; // �T�u�X�e�b�v��
    float currentTimestep = maxTimestep; // �K���I�^�C���X�e�b�v

    // ���x�ɘa�p�����[�^
    const float densityFilterCoeff = 0.1f;

    // �J�[�l���֐��̒萔�����O�v�Z
    const float poly6Coeff;
    const float spikyGradCoeff;
    const float viscosityLapCoeff;

    // ���E
    float minX = 0.0f, maxX = 10.0f;
    float minY = 0.0f, maxY = 10.0f;
    const float boundaryRadius = 0.1f; // ���E�C���^���N�V�������a

    std::vector<Particle> particles;
    SpatialHash spatialHash;

    // ���q�`��p�̃e�N�X�`��
    int particleTexture = -1;

    // �p�t�H�[�}���X����p
    std::deque<float> fpsHistory;
    std::deque<float> particleCountHistory;

    // �ߖT���̃L���b�V��
    std::vector<std::vector<std::pair<int, float>>> neighborCache;

    // SPH�J�[�l���֐�
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

    // ���ǂ��ꂽ���E�͌v�Z�֐�
    Vec2 calculateBoundaryForce(const Vec2& position, float radius) {
        Vec2 force(0, 0);

        // �������E�Ƃ̑��ݍ�p
        if (position.x < minX + radius) {
            float dist = position.x - minX;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.x += boundaryStiffness * penetration;
            }
        }

        // �E�����E�Ƃ̑��ݍ�p
        if (position.x > maxX - radius) {
            float dist = maxX - position.x;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.x -= boundaryStiffness * penetration;
            }
        }

        // �������E�Ƃ̑��ݍ�p
        if (position.y < minY + radius) {
            float dist = position.y - minY;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.y += boundaryStiffness * penetration;
            }
        }

        // �㑤���E�Ƃ̑��ݍ�p
        if (position.y > maxY - radius) {
            float dist = maxY - position.y;
            float penetration = radius - dist;
            if (penetration > 0) {
                force.y -= boundaryStiffness * penetration;
            }
        }

        return force;
    }

    // �e�N�X�`���̏�����
    void initParticleTexture() {
        // �~�`�̗��q�e�N�X�`�����쐬
        int size = static_cast<int>(particleRadius * SCREEN_WIDTH / 5.0f) * 2;
        size = std::max(size, 8); // �ŏ��T�C�Y

        particleTexture = MakeScreen(size, size, TRUE);

        SetDrawScreen(particleTexture);
        ClearDrawScreen();

        // �ڂ������ʂ����������q�̕`��
        int center = size / 2;
        float radius = size / 2.0f;

        // �O���f�[�V�����ŗ��q��`��
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

    // ���q�̒ǉ�
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

        // �L���b�V�����X�V
        neighborCache.resize(particles.size());
    }

    // �l�p�`�̈�ɗ��q���ϓ��ɔz�u
    void createParticleBlock(int rows, int cols, const Vec2& origin, float spacing) {
        // �����������O�m��
        particles.reserve(particles.size() + rows * cols);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                // ���������_������������
                float jitterX = (rand() % 100 - 50) * 0.0002f;
                float jitterY = (rand() % 100 - 50) * 0.0002f;
                Vec2 pos(origin.x + x * spacing + jitterX, origin.y + y * spacing + jitterY);
                addParticle(pos);
            }
        }

        // ��ԃn�b�V���̃T�C�Y��\��
        spatialHash.reserve(particles.size());
    }

    // �ߖT���q���̌v�Z�ƃL���b�V��
    void computeNeighbors() {
        // ��ԃn�b�V���̍X�V
        spatialHash.clear();
        for (size_t i = 0; i < particles.size(); i++) {
            spatialHash.insert(i, particles[i].position);
        }

        // �e���q�̋ߖT�����v�Z
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); i++) {
            Particle& pi = particles[i];
            auto& neighbors = neighborCache[i];
            neighbors.clear();

            // �ߖT���q���擾
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
        }
    }

    // �V�~�����[�V�����̃��C���X�e�b�v
    void update() {
        // �K���I�^�C���X�e�b�v�̌v�Z
        updateTimestep();

        // �T�u�X�e�b�v�Ő��x����
        for (int substep = 0; substep < substepCount; substep++) {
            float substepTime = currentTimestep / substepCount;

            // �X�e�b�v1: �ʒu�Ƒ��x�̋L�^
            for (auto& p : particles) {
                p.oldPosition = p.position;
                p.oldVelocity = p.velocity;
            }

            // �X�e�b�v2: �\���ʒu�̌v�Z�iexplicit Euler�j
            for (auto& p : particles) {
                p.predictedPosition = p.position + p.velocity * substepTime;
            }

            // �X�e�b�v3: �ߖT���̌v�Z�i�\���ʒu�x�[�X�j
            computeNeighbors();

            // �X�e�b�v4: ���x�ƈ��͂̌v�Z
            computeDensityPressure();

            // �X�e�b�v5: �\�ʒ��͂ƉQ�x�̌v�Z
            computeSurfaceTensionAndVorticity();

            // �X�e�b�v6: �͂̌v�Z
            computeForces();

            // �X�e�b�v7: �ʒu�Ƒ��x�̍X�V�iVelocity Verlet�@�j
            integrateVerlet(substepTime);
        }
    }

    // �K���I�^�C���X�e�b�v�̌v�Z
    void updateTimestep() {
        float maxVelocity = 0.0f;

        for (const auto& p : particles) {
            float vel = p.velocity.length();
            maxVelocity = std::max(maxVelocity, vel);
        }

        if (maxVelocity > 1e-6f) {
            // CFL�����Ɋ�Â��^�C���X�e�b�v�v�Z
            float newTimestep = courantFactor * smoothingRadius / maxVelocity;

            // �}���ȕω�������邽�߂ɃX���[�W���O
            currentTimestep = 0.9f * currentTimestep + 0.1f * newTimestep;

            // �͈͐���
            currentTimestep = std::min(maxTimestep, std::max(minTimestep, currentTimestep));
        }
        else {
            currentTimestep = maxTimestep;
        }
    }

    // ���x�ƈ��͂̌v�Z�i���ǔŁF���x�t�B���^�����O�j
    void computeDensityPressure() {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); i++) {
            Particle& pi = particles[i];
            float newDensity = 0.0f;

            // �������g�̊�^
            newDensity += massPerParticle * kernelPoly6(0, smoothingRadius);

            // �ߖT���q�̊�^
            for (const auto& [j, r] : neighborCache[i]) {
                if (i == j) continue;
                newDensity += massPerParticle * kernelPoly6(r, smoothingRadius);
            }

            // ���x�̎��ԃt�B���^�����O
            if (pi.previousDensity > 0) {
                pi.density = (1.0f - densityFilterCoeff) * pi.previousDensity + densityFilterCoeff * newDensity;
            }
            else {
                pi.density = newDensity;
            }
            pi.previousDensity = pi.density;

            // ���ǔň��͌v�Z�iTait�������j
            const float gamma = 7.0f;  // �C�̒萔�ɑ���
            const float B = pressureCoeff * restDensity / gamma;

            float ratio = pi.density / restDensity;
            pi.pressure = B * (std::pow(ratio, gamma) - 1.0f);
            pi.pressure = std::max(0.0f, pi.pressure); // ���̈��͂�h�~
        }
    }

    // �\�ʒ��͂ƉQ�x�̌v�Z
    void computeSurfaceTensionAndVorticity() {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); i++) {
            Particle& pi = particles[i];

            // �\�ʖ@���̌v�Z
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

            // �\�ʒ��͂̌v�Z�i�\�ʋȗ��ɔ��j
            float surfaceStrength = normal.length();
            if (surfaceStrength > 0.1f) {  // �\�ʗ��q�̔��肵�����l
                // �\�ʒ��͂̉e���͕\�ʗ��q�݂̂ɓK�p
                pi.surfaceTension = surfaceTensionCoeff * colorGradientSum;
            }
            else {
                pi.surfaceTension = 0.0f;
            }

            // �Q�x�̌v�Z
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
        }
    }

    // �͂̌v�Z�i���ǔŁF�\�ʒ��͂ƉQ�x��ǉ��j
    void computeForces() {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); i++) {
            Particle& pi = particles[i];
            // �d��
            pi.force = Vec2(0, gravityY * pi.density);

            // ���E��
            pi.force += calculateBoundaryForce(pi.position, boundaryRadius);

            // ���q�ԗ�
            for (const auto& [j, r] : neighborCache[i]) {
                if (i == j) continue;

                Particle& pj = particles[j];
                Vec2 rij = pi.position - pj.position;

                // ���ɋ߂��ꍇ�͒P�ʃx�N�g���𒲐�
                Vec2 rijNorm = (r > 1e-6f) ? rij * (1.0f / r) : Vec2(0, 1);

                // ���͍��i���ǔŁF�����̈��͂��g�p�j
                float pressureTerm = -massPerParticle * (pi.pressure / (pi.density * pi.density) +
                    pj.pressure / (pj.density * pj.density));
                Vec2 pressureForce = rijNorm * pressureTerm * kernelSpikyGradient(r, smoothingRadius);
                pi.force += pressureForce;

                // �S����
                Vec2 vij = pj.velocity - pi.velocity;
                float viscosityTerm = viscosityCoeff * massPerParticle / pj.density;
                Vec2 viscosityForce = vij * viscosityTerm * kernelViscosityLaplacian(r, smoothingRadius);
                pi.force += viscosityForce;

                // �\�ʒ��͍�
                if (pi.surfaceTension > 0) {
                    Vec2 surfaceForce = rijNorm * pi.surfaceTension * kernelPoly6(r, smoothingRadius);
                    pi.force += surfaceForce;
                }

                // �Q�x���i�Q�������j
                if (pi.vorticityStrength > 0) {
                    // �Q�x�ɔ�Ⴕ���͂�������i�����_���ȕ����j
                    float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f;
                    Vec2 vortexForce(std::cos(angle), std::sin(angle));
                    pi.force += vortexForce * pi.vorticityStrength * 0.01f;
                }

                // XSPH�C���i���x�̕��U��ጸ�j
                float xsphWeight = xsphCoeff * massPerParticle * kernelPoly6(r, smoothingRadius) / pj.density;
                pi.velocity += vij * xsphWeight;
            }
        }
    }

    // Velocity Verlet�@�ɂ��ϕ��i��荂���x�j
    void integrateVerlet(float dt) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(particles.size()); i++) {
            Particle& p = particles[i];

            // �����x = �� / ����
            Vec2 acceleration = p.force * (1.0f / p.density);

            // ���x�̔��X�e�b�v�X�V
            Vec2 halfVelocity = p.velocity + acceleration * (dt * 0.5f);

            // �ʒu���X�V
            p.position += halfVelocity * dt;

            // ���E����
            processBoundaryConditions(p);

            // �ŏI�I�ȑ��x�X�V
            p.velocity = halfVelocity + acceleration * (dt * 0.5f);
        }
    }

    // ���ǂ��ꂽ���E��������
    void processBoundaryConditions(Particle& p) {
        // �������E
        if (p.position.x < minX) {
            p.position.x = minX + (minX - p.position.x) * 0.5f;
            p.velocity.x = -p.velocity.x * std::exp(-boundaryDampingCoeff * fabs(p.velocity.x));
        }
        // �E�����E
        if (p.position.x > maxX) {
            p.position.x = maxX - (p.position.x - maxX) * 0.5f;
            p.velocity.x = -p.velocity.x * std::exp(-boundaryDampingCoeff * fabs(p.velocity.x));
        }
        // �������E
        if (p.position.y < minY) {
            p.position.y = minY + (minY - p.position.y) * 0.5f;
            p.velocity.y = -p.velocity.y * std::exp(-boundaryDampingCoeff * fabs(p.velocity.y));
        }
        // �㑤���E
        if (p.position.y > maxY) {
            p.position.y = maxY - (p.position.y - maxY) * 0.5f;
            p.velocity.y = -p.velocity.y * std::exp(-boundaryDampingCoeff * fabs(p.velocity.y));
        }
    }

    // �����_�����O�p�ɗ��q�̈ʒu���擾
    const std::vector<Particle>& getParticles() const {
        return particles;
    }

    // �V�~�����[�V�����̈�̐ݒ�
    void setBoundary(float _minX, float _maxX, float _minY, float _maxY) {
        minX = _minX;
        maxX = _maxX;
        minY = _minY;
        maxY = _maxY;
    }

    // �V�~�����[�V�����̈��`�悷��
    void drawBoundary() {
        // �V�~�����[�V�������W���X�N���[�����W�ɕϊ�
        int x1 = static_cast<int>(minX * SCREEN_WIDTH / 10.0f);
        int y1 = SCREEN_HEIGHT - static_cast<int>(maxY * SCREEN_HEIGHT / 10.0f);
        int x2 = static_cast<int>(maxX * SCREEN_WIDTH / 10.0f);
        int y2 = SCREEN_HEIGHT - static_cast<int>(minY * SCREEN_HEIGHT / 10.0f);

        // ���E�𔒂��l�p�ŕ`��
        DrawBox(x1, y1, x2, y2, GetColor(255, 255, 255), FALSE);
    }

    // ���q��`�悷�� - �e�N�X�`���g�p�̍œK����
    void drawParticles() {
        int textureSize = GetDrawFormatStringWidth("%d", particleTexture);
        float scaleFactor = particleRadius * SCREEN_WIDTH / 5.0f / (textureSize / 2);

        // �J�����O�p�̉�ʔ͈�
        int screenLeft = -textureSize;
        int screenTop = -textureSize;
        int screenRight = SCREEN_WIDTH + textureSize;
        int screenBottom = SCREEN_HEIGHT + textureSize;

        SetDrawBlendMode(DX_BLENDMODE_ALPHA, 255);

        for (const auto& p : particles) {
            // �V�~�����[�V�������W���X�N���[�����W�ɕϊ�
            int x = static_cast<int>(p.position.x * SCREEN_WIDTH / 10.0f);
            int y = SCREEN_HEIGHT - static_cast<int>(p.position.y * SCREEN_HEIGHT / 10.0f);

            // ��ʊO�̗��q�̓X�L�b�v
            if (x < screenLeft || x > screenRight || y < screenTop || y > screenBottom) {
                continue;
            }

            // ���x�ƕ\�ʒ��͂ɉ����ăJ���[�𒲐�
            float densityFactor = std::min(1.0f, p.density / (restDensity * 2.0f));
            float surfaceFactor = p.surfaceTension * 10.0f;

            int r = static_cast<int>(64 + 191 * densityFactor);
            int g = static_cast<int>(164 - 100 * densityFactor + 100 * surfaceFactor);
            int b = static_cast<int>(255 - 100 * densityFactor);
            int particleColor = GetColor(r, g, b);

            // �e�N�X�`����`��
            DrawRotaGraph(x, y, scaleFactor, 0, particleTexture, TRUE);
        }

        SetDrawBlendMode(DX_BLENDMODE_NOBLEND, 255);
    }

    // �p�t�H�[�}���X����`��
    void drawPerformanceInfo() {
        // ���݂�FPS���v�Z���A�����ɒǉ�
        float currentFPS = GetFPS();
        fpsHistory.push_back(currentFPS);
        if (fpsHistory.size() > 60) {
            fpsHistory.pop_front();
        }

        // ���q�������̍X�V
        particleCountHistory.push_back(static_cast<float>(particles.size()));
        if (particleCountHistory.size() > 60) {
            particleCountHistory.pop_front();
        }

        // ����FPS�̌v�Z
        float avgFPS = 0.0f;
        for (float fps : fpsHistory) {
            avgFPS += fps;
        }
        avgFPS /= fpsHistory.size();

        // ���\��
        DrawFormatString(10, 10, GetColor(255, 255, 255), "FPS: %.1f", avgFPS);
        DrawFormatString(10, 30, GetColor(255, 255, 255), "Particles: %d", particles.size());
        DrawFormatString(10, 50, GetColor(255, 255, 255), "Timestep: %.4f", currentTimestep);
    }

    // �}�E�X�ʒu�ɗ��q��ǉ�
    void addParticlesAtMouse(int num) {
        int mouseX, mouseY;
        GetMousePoint(&mouseX, &mouseY);

        // �}�E�X�ʒu���V�~�����[�V�������W�ɕϊ�
        float simX = mouseX * 10.0f / SCREEN_WIDTH;
        float simY = (SCREEN_HEIGHT - mouseY) * 10.0f / SCREEN_HEIGHT;

        // �}�E�X�ʒu���ӂɃ����_���ɗ��q��ǉ�
        for (int i = 0; i < num; i++) {
            float offsetX = (rand() % 100 - 50) * 0.01f;
            float offsetY = (rand() % 100 - 50) * 0.01f;
            Vec2 pos(simX + offsetX, simY + offsetY);
            Vec2 vel(offsetX * 2.0f, offsetY * 2.0f);
            addParticle(pos, vel);
        }
    }

    // �_������V�i���I���Z�b�g�A�b�v
    void setupDamBreakScenario() {
        // �����̗��q���N���A
        particles.clear();
        neighborCache.clear();

        // �����ɐ��̉���쐬
        float spacing = particleRadius * 1.5f;
        int rows = static_cast<int>((maxY - minY) * 0.8f / spacing);
        int cols = static_cast<int>((maxX - minX) * 0.4f / spacing);

        Vec2 origin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, origin, spacing);
    }

    // ���H�����V�i���I���Z�b�g�A�b�v
    void setupDropScenario() {
        // �����̗��q���N���A
        particles.clear();
        neighborCache.clear();

        // ���̒r���쐬�i�����ɏ��ʂ̐��j
        float spacing = particleRadius * 1.5f;
        int rows = 3;
        int cols = static_cast<int>((maxX - minX) * 0.8f / spacing);

        Vec2 origin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, origin, spacing);

        // �㕔�ɐ��H���쐬
        int dropRadius = 5;
        Vec2 dropCenter(maxX * 0.5f, maxY * 0.8f);

        for (int y = -dropRadius; y <= dropRadius; y++) {
            for (int x = -dropRadius; x <= dropRadius; x++) {
                float dist = std::sqrt(x * x + y * y);
                if (dist <= dropRadius) {
                    Vec2 pos(dropCenter.x + x * spacing, dropCenter.y + y * spacing);
                    addParticle(pos, Vec2(0, -5.0f)); // �������̏����x��^����
                }
            }
        }
    }

    // �񗬑̍����V�i���I���Z�b�g�A�b�v
    void setupTwoFluidScenario() {
        // �����̗��q���N���A
        particles.clear();
        neighborCache.clear();

        // �����ƉE���ɂ��ꂼ��قȂ閧�x�̗��̂�z�u
        float spacing = particleRadius * 1.5f;
        int rows = static_cast<int>((maxY - minY) * 0.6f / spacing);
        int cols = static_cast<int>((maxX - minX) * 0.4f / spacing) / 2;

        // �����̗��́i�����x�j
        Vec2 leftOrigin(minX + spacing, minY + spacing);
        createParticleBlock(rows, cols, leftOrigin, spacing);

        // �E���̗��́i�ᖧ�x�j
        Vec2 rightOrigin(maxX - cols * spacing, minY + spacing);
        int rightParticleStart = particles.size();
        createParticleBlock(rows, cols, rightOrigin, spacing);

        // �E���̗��q�͈قȂ���������悤�ɐݒ�
        for (size_t i = rightParticleStart; i < particles.size(); i++) {
            particles[i].density = restDensity * 0.7f; // �ᖧ�x
            particles[i].previousDensity = restDensity * 0.7f;
        }
    }
};

// ���C���֐�
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // DX���C�u�����̏�����
    SetGraphMode(SCREEN_WIDTH, SCREEN_HEIGHT, 32);
    ChangeWindowMode(TRUE);
    SetWindowText("Improved SPH Fluid Simulation");
    if (DxLib_Init() == -1) return -1;
    SetDrawScreen(DX_SCREEN_BACK);

    // OpenMP�̏�����
    omp_set_num_threads(std::max(1, omp_get_max_threads()));

    // �V�~�����[�^�̍쐬
    ImprovedSPHFluidSimulator simulator;
    simulator.setBoundary(0.5f, 9.5f, 0.5f, 9.5f);
    simulator.setupDamBreakScenario();

    // �V�i���I�I��p�̕ϐ�
    int currentScenario = 0; // 0: �_������, 1: ���H, 2: �񗬑�

    int frameCount = 0;
    bool isRunning = true;
    bool isPaused = false;

    // ���C�����[�v
    while (isRunning && ProcessMessage() == 0) {
        ClearDrawScreen();

        // �L�[���͏���
        if (CheckHitKey(KEY_INPUT_ESCAPE)) isRunning = false;
        if (CheckHitKey(KEY_INPUT_SPACE) && frameCount % 15 == 0) isPaused = !isPaused;

        // �}�E�X���͏���
        if ((GetMouseInput() & MOUSE_INPUT_LEFT) != 0 && frameCount % 5 == 0) {
            simulator.addParticlesAtMouse(5);
        }

        // �V�i���I�؂�ւ�
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

        // �V�~�����[�V�����X�V
        if (!isPaused) {
            simulator.update();
        }

        // �`��
        simulator.drawBoundary();
        simulator.drawParticles();
        simulator.drawPerformanceInfo();

        // �V�i���I���\��
        const char* scenarioNames[] = { "Dam Break", "Droplet", "Two Fluids" };
        DrawFormatString(10, 70, GetColor(255, 255, 255), "Scenario: %s", scenarioNames[currentScenario]);
        DrawString(10, SCREEN_HEIGHT - 60, "Controls:", GetColor(200, 200, 200));
        DrawString(10, SCREEN_HEIGHT - 40, "1-3: Change Scenario, Space: Pause, Left Click: Add Particles", GetColor(200, 200, 200));
        DrawString(10, SCREEN_HEIGHT - 20, "Esc: Exit", GetColor(200, 200, 200));

        // ��ʂ̍X�V
        ScreenFlip();
        frameCount++;
    }

    DxLib_End();
    return 0;
}