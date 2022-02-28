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

#include "stdafx.h"
#include "OpenGL.h"

/////////////////////////////////////////////////////////////////////////////
// Constructors

// All Setup For OpenGL Goes Here
CGL::CGL()
{
	m_bFullScreen	= true;
	m_hDC			= NULL;		  // Private GDI Device Context
	m_hRC			= NULL;		  // Permanent Rendering Context
	m_hWnd			= NULL;		  // Holds Our Window Handle
}

/*	This Code Creates Our OpenGL Window.  Parameters Are:
 *	lpszTitle		- Title to Appear at the Top of the Window
 *	hWndParent		- Parent HWND where the screne will be drawn
 *	nZBuffer		- Number of Bits to Use for Z-Buffer (16/32)
 */
bool CGL::Create(char* lpszTitle, HWND hWndParent, int nZBuffer=16) : m_bFullScreen(false)
{
	GLuint		PixelFormat;							// Holds The Results After Searching For A Match
	WNDCLASS	wc;										// Windows Class Structure
	DWORD		dwExStyle;								// Window Extended Style
	DWORD		dwStyle;								// Window Style
	RECT		rectWindow;								// Grabs Rectangle Upper Left / Lower Right Values
	int			nWidth,
				nHeight;

	// Get window size
	GetClientRect(hWndParent, &rectWindow);
	nWidth  = rectWindow.right-rectWindow.left;
	nHeight = rectWindow.bottom-rectWindow.top;

	m_hInstance			= GetModuleHandle(NULL);		// Grab An Instance For Our Window
	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Redraw On Size, And Own DC For Window.
	wc.lpfnWndProc		= (WNDPROC) WndProc;			// WndProc Handles Messages
	wc.cbClsExtra		= 0;							// No Extra Window Data
	wc.cbWndExtra		= 0;							// No Extra Window Data
	wc.hInstance		= m_hInstance;					// Set The Instance
	wc.hIcon			= LoadIcon(NULL, IDI_WINLOGO);	// Load The Default Icon
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);	// Load The Arrow Pointer
	wc.hbrBackground	= NULL;							// No Background Required For GL
	wc.lpszMenuName		= NULL;							// We Don't Want A Menu
	wc.lpszClassName	= "OpenGL";						// Set The Class Name

	if (!RegisterClass(&wc))							// Attempt To Register The Window Class
	{
		MessageBox(m_hWnd,"Failed To Register The Window Class.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}
	
	dwExStyle = WS_EX_APPWINDOW;						// Window Extended Style
	//dwStyle = WS_OVERLAPPEDWINDOW;					// Windows Style
	dwStyle = WS_DISABLED | WS_VISIBLE | WS_CHILD;		// Windows Style

	AdjustWindowRectEx(&rectWindow, dwStyle, false, dwExStyle);		// Adjust Window To true Requested Size

	// Create The Window
	if (!(m_hWnd=CreateWindowEx(dwExStyle,				// Extended Style For The Window
								"OpenGL",				// Class Name
								lpszTitle,				// Window Title
								dwStyle |				// Defined Window Style
								WS_CLIPSIBLINGS |		// Required Window Style
								WS_CLIPCHILDREN,		// Required Window Style
								0, 0,					// Window Position
								WindowRect.right-WindowRect.left,	// Calculate Window Width
								WindowRect.bottom-WindowRect.top,	// Calculate Window Height
								NULL,					// No Parent Window
								NULL,					// No Menu
								m_hInstance,			// Instance
								NULL)))					// Dont Pass Anything To WM_CREATE
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Window Creation Error.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	static	PIXELFORMATDESCRIPTOR pfd =					// pfd Tells Windows How We Want Things To Be
	{
		sizeof(PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
		1,												// Version Number
		PFD_DRAW_TO_WINDOW |							// Format Must Support Window
		PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
		PFD_DOUBLEBUFFER,								// Must Support Double Buffering
		PFD_TYPE_RGBA,									// Request An RGBA Format
		nBits,											// Select Our Color Depth
		0, 0, 0, 0, 0, 0,								// Color Bits Ignored
		0,												// No Alpha Buffer
		0,												// Shift Bit Ignored
		0,												// No Accumulation Buffer
		0, 0, 0, 0,										// Accumulation Bits Ignored
		nZBuffer,										// 16/32Bit Z-Buffer (Depth Buffer)  
		0,												// No Stencil Buffer
		0,												// No Auxiliary Buffer
		PFD_MAIN_PLANE,									// Main Drawing Layer
		0,												// Reserved
		0, 0, 0											// Layer Masks Ignored
	};
	
	if (!(m_hDC=GetDC(m_hWnd)))							// Did We Get A Device Context?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Create A GL Device Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if (!(PixelFormat=ChoosePixelFormat(m_hDC,&pfd)))	// Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Find A Suitable PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if(!SetPixelFormat(m_hDC,PixelFormat,&pfd))			// Are We Able To Set The Pixel Format?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Set The PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if (!(hRC=wglCreateContext(m_hDC)))					// Are We Able To Get A Rendering Context?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Create A GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if(!wglMakeCurrent(m_hDC,hRC))						// Try To Activate The Rendering Context
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Activate The GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	ShowWindow(m_hWnd, SW_SHOW);						// Show The Window
	SetForegroundWindow(m_hWnd);						// Slightly Higher Priority
	SetFocus(m_hWnd);									// Sets Keyboard Focus To The Window
	Resize(nWidth, nHeight);							// Set Up Our Perspective GL Screen

	if (!Init())										// Initialize Our Newly Created GL Window
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Initialization Failed.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	return true;										// Success
}

/*	This Code Creates Our OpenGL Window.  Parameters Are:
 *	lpszTitle		- Title to Appear at the Top of the Window
 *	nWidth			- Width of the GL Window or Fullscreen Mode
 *	nHeight			- Height of the GL Window or Fullscreen Mode
 *	nBits			- Number of Bits to Use for Color (8/16/24/32)
 *	nZBuffer		- Number of Bits to Use for Z-Buffer (16/32)
 *	bFullScreen		- Use Fullscreen Mode (true) Or Windowed Mode (false)
 */
bool CGL::Create(char* lpszTitle, int nWidth, int nHeight, int nBits=16, int nZBuffer=16, bool bFullScreen=true) : m_bFullScreen(bFullScreen)
{
	GLuint		PixelFormat;							// Holds The Results After Searching For A Match
	WNDCLASS	wc;										// Windows Class Structure
	DWORD		dwExStyle;								// Window Extended Style
	DWORD		dwStyle;								// Window Style
	RECT		rectWindow(0, nWidth, 0, nHeight);		// Grabs Rectangle Upper Left / Lower Right Values

	m_hInstance			= GetModuleHandle(NULL);		// Grab An Instance For Our Window
	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Redraw On Size, And Own DC For Window.
	wc.lpfnWndProc		= (WNDPROC) WndProc;			// WndProc Handles Messages
	wc.cbClsExtra		= 0;							// No Extra Window Data
	wc.cbWndExtra		= 0;							// No Extra Window Data
	wc.hInstance		= m_hInstance;					// Set The Instance
	wc.hIcon			= LoadIcon(NULL, IDI_WINLOGO);	// Load The Default Icon
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);	// Load The Arrow Pointer
	wc.hbrBackground	= NULL;							// No Background Required For GL
	wc.lpszMenuName		= NULL;							// We Don't Want A Menu
	wc.lpszClassName	= "OpenGL";						// Set The Class Name

	if (!RegisterClass(&wc))							// Attempt To Register The Window Class
	{
		MessageBox(m_hWnd,"Failed To Register The Window Class.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}
	
	if (m_bFullScreen)									// Attempt Fullscreen Mode?
	{
		DEVMODE dmScreenSettings;						// Device Mode
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));	// Makes Sure Memory's Cleared
		dmScreenSettings.dmSize		 = sizeof(dmScreenSettings);	// Size Of The Devmode Structure
		dmScreenSettings.dmPelsWidth  = nWidth;			// Selected Screen Width
		dmScreenSettings.dmPelsHeight = nHeight;		// Selected Screen Height
		dmScreenSettings.dmBitsPerPel = nBits;			// Selected Bits Per Pixel
		dmScreenSettings.dmFields	  = DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		// Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
		if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
		{
			// If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
			if (MessageBox(m_hWnd,"The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?","NeHe GL",MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
				m_bFullScreen = false;					// Windowed Mode Selected.  Fullscreen = false
			else
			{
				// Pop Up A Message Box Letting User Know The Program Is Closing.
				MessageBox(m_hWnd,"Program Will Now Close.","ERROR",MB_OK|MB_ICONSTOP);
				return false;							// Return false
			}
		}
	}

	if (m_bFullScreen)									// Are We Still In Fullscreen Mode?
	{
		dwExStyle = WS_EX_APPWINDOW;					// Window Extended Style
		dwStyle = WS_POPUP;								// Windows Style
		ShowCursor(false);								// Hide Mouse Pointer
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;	// Window Extended Style
		dwStyle = WS_OVERLAPPEDWINDOW;					// Windows Style
	}

	AdjustWindowRectEx(&rectWindow, dwStyle, false, dwExStyle);		// Adjust Window To true Requested Size

	// Create The Window
	if (!(m_hWnd=CreateWindowEx(dwExStyle,				// Extended Style For The Window
								"OpenGL",				// Class Name
								lpszTitle,				// Window Title
								dwStyle |				// Defined Window Style
								WS_CLIPSIBLINGS |		// Required Window Style
								WS_CLIPCHILDREN,		// Required Window Style
								0, 0,					// Window Position
								WindowRect.right-WindowRect.left,	// Calculate Window Width
								WindowRect.bottom-WindowRect.top,	// Calculate Window Height
								NULL,					// No Parent Window
								NULL,					// No Menu
								m_hInstance,			// Instance
								NULL)))					// Dont Pass Anything To WM_CREATE
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Window Creation Error.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	static	PIXELFORMATDESCRIPTOR pfd =					// pfd Tells Windows How We Want Things To Be
	{
		sizeof(PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
		1,												// Version Number
		PFD_DRAW_TO_WINDOW |							// Format Must Support Window
		PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
		PFD_DOUBLEBUFFER,								// Must Support Double Buffering
		PFD_TYPE_RGBA,									// Request An RGBA Format
		nBits,											// Select Our Color Depth
		0, 0, 0, 0, 0, 0,								// Color Bits Ignored
		0,												// No Alpha Buffer
		0,												// Shift Bit Ignored
		0,												// No Accumulation Buffer
		0, 0, 0, 0,										// Accumulation Bits Ignored
		nZBuffer,										// 16/32Bit Z-Buffer (Depth Buffer)  
		0,												// No Stencil Buffer
		0,												// No Auxiliary Buffer
		PFD_MAIN_PLANE,									// Main Drawing Layer
		0,												// Reserved
		0, 0, 0											// Layer Masks Ignored
	};
	
	if (!(m_hDC=GetDC(m_hWnd)))							// Did We Get A Device Context?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Create A GL Device Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if (!(PixelFormat=ChoosePixelFormat(m_hDC,&pfd)))	// Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Find A Suitable PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if(!SetPixelFormat(m_hDC,PixelFormat,&pfd))			// Are We Able To Set The Pixel Format?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Set The PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if (!(hRC=wglCreateContext(m_hDC)))					// Are We Able To Get A Rendering Context?
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Create A GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	if(!wglMakeCurrent(m_hDC,hRC))						// Try To Activate The Rendering Context
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Can't Activate The GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	ShowWindow(m_hWnd, SW_SHOW);						// Show The Window
	SetForegroundWindow(m_hWnd);						// Slightly Higher Priority
	SetFocus(m_hWnd);									// Sets Keyboard Focus To The Window
	Resize(nWidth, nHeight);							// Set Up Our Perspective GL Screen

	if (!Init())										// Initialize Our Newly Created GL Window
	{
		KillGLWindow();									// Reset The Display
		MessageBox(m_hWnd,"Initialization Failed.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;									// Return false
	}

	return true;										// Success
}

// Properly Kill The Window
void CGL::Destroy()
{
	if (m_bFullScreen)									// Are We In Fullscreen Mode?
	{
		ChangeDisplaySettings(NULL,0);					// If So Switch Back To The Desktop
		ShowCursor(true);								// Show Mouse Pointer
	}

	if (m_hRC)											// Do We Have A Rendering Context?
	{
		if (!wglMakeCurrent(NULL, NULL))				// Are We Able To Release The DC And RC Contexts?
			MessageBox(m_hWnd,"Release Of DC And RC Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);

		if (!wglDeleteContext(hRC))						// Are We Able To Delete The RC?
			MessageBox(m_hWnd,"Release Rendering Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		m_hRC=NULL;										// Set RC To NULL
	}

	if (m_hDC && !ReleaseDC(m_hWnd, m_hDC))				// Are We Able To Release The DC
	{
		MessageBox(m_hWnd,"Release Device Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		m_hDC=NULL;										// Set DC To NULL
	}

	if (m_hWnd && !DestroyWindow(m_hWnd))				// Are We Able To Destroy The Window?
	{
		m_hWnd=NULL;									// Set m_hWnd To NULL
		MessageBox(m_hWnd,"Could Not Release m_hWnd.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
	}

	if (!UnregisterClass("OpenGL", m_hInstance))		// Are We Able To Unregister Class
	{
		MessageBox(m_hWnd,"Could Not Unregister Class.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		m_hInstance=NULL;								// Set hInstance To NULL
	}
}

/////////////////////////////////////////////////////////////////////////////
// Attributes

void CGL::SetFullScreen(bool bFullScreen)
{
	m_bFullScreen = bFullScreen;
}

bool CGL::GetFullScreen() const
{
	return m_bFullScreen;
}

HDC CGL::GethDC() const
{
	return m_hDC;
}

HWND CGL::GethWnd() const
{
	return m_hWnd;
}

/////////////////////////////////////////////////////////////////////////////
// Operations

bool CGL::Init()
{
	return OnInit();
}

// Resize And Initialize The GL Window
bool CGL::Resize(GLsizei nWidth, GLsizei nHeight)
{
	if (nWidth < 0 || nHeight < 0)
		return false;
	if (nHeight == 0)									// Prevent A Divide By Zero By
		nHeight = 1;									// Making Height Equal One

	glViewport(0, 0, nWidth, nHeight);					// Reset The Current Viewport

	return OnResize();
}

bool CGL::Render()
{
	return OnRender();
}

/////////////////////////////////////////////////////////////////////////////
// Overridables

bool CGL::OnInit()
{
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

	return true;
}

bool CGL::OnResize(GLsizei nWidth, GLsizei nHeight)
{
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f, (GLfloat)nWidth/(GLfloat)nHeight, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
	return true;
}

bool CGL::OnRender()
{
	static float	fXRot = 0.0f,
					fYRot = 0.0f,
					fZRot = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer
	glLoadIdentity();									// Reset The View
	glTranslatef(0.0f, 0.0f, -5.0f);

	glRotatef(fXRot, 1.0f, 0.0f, 0.0f);
	glRotatef(fYRot, 0.0f, 1.0f, 0.0f);
	glRotatef(fZRot, 0.0f, 0.0f, 1.0f);

	fXRot += 0.3f;
	fYRot += 0.2f;
	fZRot += 0.4f;

	glBegin(GL_QUADS);
		// Front Face
		glColor3f(1.0f, 0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
		glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
		glColor3f(1.0f, 1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
		glColor3f(1.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
		// Back Face
		glColor3f(0.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
		glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
		glColor3f(0.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
		glColor3f(0.0f, 0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
		// Top Face
		glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
		glColor3f(1.0f, 0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
		glColor3f(1.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
		glColor3f(0.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
		// Bottom Face
		glColor3f(0.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
		glColor3f(0.0f, 1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
		glColor3f(1.0f, 1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
		glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
		// Right face
		glColor3f(0.0f, 1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
		glColor3f(0.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
		glColor3f(1.0f, 1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
		glColor3f(1.0f, 1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
		// Left Face
		glColor3f(0.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
		glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
		glColor3f(1.0f, 0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
		glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
	glEnd();

	return true;
}
