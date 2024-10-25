#include <windows.h>
#include <bits/stdc++.h>

void move(int x, int y) {
	SetCursorPos(x, y);
}

void click() {
	INPUT input = {0};
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
	SendInput(1, &input, sizeof(INPUT));
	SendInput(1, &input, sizeof(INPUT));

	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
	SendInput(1, &input, sizeof(INPUT));
}

std::pair<double, double> randNoise() {
	
}

int main () {
	int x, y;
	std::cin >> x >> y;
	move(x, y);
	Sleep(600);
	click();
	return 0;
}