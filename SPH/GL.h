/* 
   Mortar Terror
   Copyright (C) 2000-2002 Werner BEROUX
   
   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#pragma once

class CGL
{
public:
// Constructors
	CGL();
	bool Create(char* lpszTitle, HWND hWndParent, int nZBuffer);
	bool Create(char* lpszTitle, int nWidth, int nHeight, int nBits, int nZBuffer, bool bFullScreen);
	void Destroy() ;

// Attributes
	void SetFullScreen(bool bFullScreen);
	bool GetFullScreen() const;
	HDC GethDC() const;
	HWND GethWnd() const;

// Operations
	bool Init();
	bool Resize(GLsizei nWidth, GLsizei nHeight);
	bool Render();
	friend LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

protected:
// Overridables
	virtual bool OnInit();
	virtual bool OnResize(GLsizei nWidth, GLsizei nHeight);
	virtual bool OnRender();

// Implementation
	bool      m_bFullScreen;	// Fullscreen Flag Set To Fullscreen Mode By Default
	HDC       m_hDC;			// Private GDI Device Context
	HGLRC     m_hRC;			// Permanent Rendering Context
	HWND      m_hWnd;			// Holds Our Window Handle
	HINSTANCE m_hInstance;		// Holds The Instance Of The Application
};