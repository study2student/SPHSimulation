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

// 線分とカプセルの当たり判定
bool LineSegmentCapsuleIntersection(VECTOR lineStart, VECTOR lineEnd, VECTOR capsuleStart, VECTOR capsuleEnd, float capsuleRadius) {
    // 線分をベクトルとして表現
    VECTOR lineVector = VSub(lineEnd, lineStart);

    // カプセルの軸ベクトルと中心線のベクトル
    VECTOR capsuleAxis = VSub(capsuleEnd, capsuleStart);
    VECTOR capsuleCenterLine = VSub(lineStart, capsuleStart);

    // 線分とカプセルの中心線の最近点を求める
    VECTOR closestPoint = ClosestPointOnLineSegment(capsuleCenterLine, VECTOR{ 0, 0, 0 }, capsuleAxis);

    // 最近点がカプセルの中心線の範囲内にあるかどうかを判定
    if (VDot(capsuleCenterLine, closestPoint) <= capsuleRadius) {
        // 線分上の最近点
        VECTOR closestPointOnLine = VAdd(closestPoint, capsuleStart);

        // 線分上の最近点が線分内にあるかどうかを判定
        if (VDot(VSub(lineStart, closestPointOnLine), VSub(lineEnd, closestPointOnLine)) <= 0) {
            return true; // 当たっている
        }
    }

    return false; // 当たっていない
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // 判定する線分とカプセルの座標を設定
    VECTOR lineStart = { 100, 100, 0 };
    VECTOR lineEnd = { 200, 200, 0 };

    VECTOR capsuleStart = { 150, 150, 0 };
    VECTOR capsuleEnd = { 250, 250, 0 };
    float capsuleRadius = 20.0f;

    // 線分とカプセルの当たり判定を実行
    bool isIntersecting = LineSegmentCapsuleIntersection(lineStart, lineEnd, capsuleStart, capsuleEnd, capsuleRadius);

    if (isIntersecting) {
        // 当たっている場合の処理
        // ここに処理を記述
        DrawString(0, 0, "hellow", 0x00ff00, 0xffffff);
    }
    else {
        // 当たっていない場合の処理
        // ここに処理を記述
    }

    //    // 裏画面の内容を表画面に反映
    //    ScreenFlip();

    //// ＤＸライブラリの後始末
    //DxLib_End();

    // ソフトの終了
    return 0;
}
