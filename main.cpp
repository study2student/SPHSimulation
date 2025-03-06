#include <DxLib.h>

VECTOR ClosestPointOnLineSegment(VECTOR point, VECTOR lineStart, VECTOR lineEnd) {
    VECTOR lineDirection = VSub(lineEnd, lineStart);
    float lineLength = VSize(lineDirection);
    lineDirection = VScale(lineDirection, 1.0f / lineLength);

    VECTOR pointToLineStart = VSub(point, lineStart);
    float t = VDot(pointToLineStart, lineDirection);

    if (t <= 0.0f) {
        return lineStart;
    }
    if (t >= lineLength) {
        return lineEnd;
    }

    return VAdd(lineStart, VScale(lineDirection, t));
}

// �����ƃJ�v�Z���̓����蔻��
bool LineSegmentCapsuleIntersection(VECTOR lineStart, VECTOR lineEnd, VECTOR capsuleStart, VECTOR capsuleEnd, float capsuleRadius) {
    // �������x�N�g���Ƃ��ĕ\��
    VECTOR lineVector = VSub(lineEnd, lineStart);

    // �J�v�Z���̎��x�N�g���ƒ��S���̃x�N�g��
    VECTOR capsuleAxis = VSub(capsuleEnd, capsuleStart);
    VECTOR capsuleCenterLine = VSub(lineStart, capsuleStart);

    // �����ƃJ�v�Z���̒��S���̍ŋߓ_�����߂�
    VECTOR closestPoint = ClosestPointOnLineSegment(capsuleCenterLine, VECTOR{ 0, 0, 0 }, capsuleAxis);

    // �ŋߓ_���J�v�Z���̒��S���͈͓̔��ɂ��邩�ǂ����𔻒�
    if (VDot(capsuleCenterLine, closestPoint) <= capsuleRadius) {
        // ������̍ŋߓ_
        VECTOR closestPointOnLine = VAdd(closestPoint, capsuleStart);

        // ������̍ŋߓ_���������ɂ��邩�ǂ����𔻒�
        if (VDot(VSub(lineStart, closestPointOnLine), VSub(lineEnd, closestPointOnLine)) <= 0) {
            return true; // �������Ă���
        }
    }

    return false; // �������Ă��Ȃ�
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // ���肷������ƃJ�v�Z���̍��W��ݒ�
    VECTOR lineStart = { 100, 100, 0 };
    VECTOR lineEnd = { 200, 200, 0 };

    VECTOR capsuleStart = { 150, 150, 0 };
    VECTOR capsuleEnd = { 250, 250, 0 };
    float capsuleRadius = 20.0f;

    // �����ƃJ�v�Z���̓����蔻������s
    bool isIntersecting = LineSegmentCapsuleIntersection(lineStart, lineEnd, capsuleStart, capsuleEnd, capsuleRadius);

    if (isIntersecting) {
        // �������Ă���ꍇ�̏���
        // �����ɏ������L�q
        DrawString(0, 0, "hellow", 0x00ff00, 0xffffff);
    }
    else {
        // �������Ă��Ȃ��ꍇ�̏���
        // �����ɏ������L�q
    }

    //    // ����ʂ̓��e��\��ʂɔ��f
    //    ScreenFlip();

    //// �c�w���C�u�����̌�n��
    //DxLib_End();

    // �\�t�g�̏I��
    return 0;
}
