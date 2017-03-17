#ifndef GLUT_DISABLE_ATEXIT_HACK  
#define GLUT_DISABLE_ATEXIT_HACK 
#endif

#define GLEW_STATIC

#include <cstdio>
#include <cstdlib>
#include <string>

#include <GL/glew.h>
#include <GL/glut.h>
//#include <glm/glm.hpp>

#include "pic.h"
#include "SimBox.h"

#pragma comment( lib, "glew32s.lib" )

using namespace std;

// window parameters
int windowWidth;
int windowHeight;

// camera parameters
double Theta = PI / 6;
double Phi = PI / 6;
double R = 3;

// mouse control
int g_iMenuId;
int g_vMousePos[2];
int g_iLeftMouseButton, g_iMiddleMouseButton, g_iRightMouseButton;

// control variables
int pause = 0; // used to control pause: 0 or resume: 1
int saveImage = 0; // save scene to image. save: 1, unsave: 0

int imageNum = 0; // number of images saved to disk so far

// shaders
const char * VERTEX_SHADER = "";
const char * FRAG_SHADER = "";

void myinit() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, 1, 1, 100);
    glLoadIdentity();

    glEnable(GL_DEPTH_TEST);

	//*
	glShadeModel(GL_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);
	glEnable(GL_LINE_SMOOTH);

	// global ambient light
	GLfloat aGa[] = { 1.0, 1.0, 1.0, 1.0 };

	// light 's ambient, diffuse, specular
	GLfloat lKa0[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat lKd0[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat lKs0[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat lP0[] = { 2.0, 2.0, 2.0, 1.0 };

	GLfloat mKa[] = { 0.5, 0.5, 0.5, 1.0 };
	GLfloat mKd[] = { 0.5, 0.5, 0.5, 1.0 };
	GLfloat mKs[] = { 0.0, 0.0, 0.0, 1.0 };

	//glMaterialfv(GL_FRONT, GL_AMBIENT, mKa);
	//glMaterialfv(GL_FRONT, GL_DIFFUSE, mKd);
	//glMaterialfv(GL_FRONT, GL_SPECULAR, mKs);

	// set up lighting 
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, aGa);
	glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

	glEnable(GL_LIGHTING);

	// macro to set up light i
#define LIGHTSETUP(i)\
    glLightfv(GL_LIGHT##i, GL_POSITION, lP##i);\
    glLightfv(GL_LIGHT##i, GL_AMBIENT, lKa##i);\
    glLightfv(GL_LIGHT##i, GL_DIFFUSE, lKd##i);\
    glLightfv(GL_LIGHT##i, GL_SPECULAR, lKs##i);\
    glEnable(GL_LIGHT##i)

	LIGHTSETUP(0);
	//*/
}

// Draw a bounding box for this fluid
void showBoundingBox() {
    glColor4f(0.6, 0.6, 0.6, 0);

    glBegin(GL_LINES);

	for (double i = -0.5 * BOUNDARY_X; i <= 0.5 * BOUNDARY_X; i += BOUNDARY_X) {
		for (double j = -0.5 * BOUNDARY_Y; j <= 0.5 * BOUNDARY_Y; j += BOUNDARY_Y) {
			glVertex3f(i, j, 0);
			glVertex3f(i, j, BOUNDARY_Z);
		}
    }

	for (double j = 0; j <= BOUNDARY_Z; j += BOUNDARY_Z) {
		glVertex3f(-0.5 * BOUNDARY_X, -0.5 * BOUNDARY_Y, j);
		glVertex3f(-0.5 * BOUNDARY_X, 0.5 * BOUNDARY_Y, j);

		glVertex3f(-0.5 * BOUNDARY_X, 0.5 * BOUNDARY_Y, j);
		glVertex3f(0.5 * BOUNDARY_X, 0.5 * BOUNDARY_Y, j);

		glVertex3f(0.5 * BOUNDARY_X, 0.5 * BOUNDARY_Y, j);
		glVertex3f(0.5 * BOUNDARY_X, -0.5 * BOUNDARY_Y, j);

		glVertex3f(0.5 * BOUNDARY_X, -0.5 * BOUNDARY_Y, j);
		glVertex3f(-0.5 * BOUNDARY_X, -0.5 * BOUNDARY_Y, j);
	}

    glEnd();
}

void display() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(R * cos(Phi) * cos(Theta), R * sin(Phi) * cos(Theta), R * sin(Theta),
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    showBoundingBox();

	static SimBox simBox;

	//*
	static int time = 0;
	if (time > 1000 && time % 10 == 0)
	{
		for (int i = 0; i < 2; ++i)
			for (int j = 1; j < 3; ++j)
			{
				simBox.put_in_particle(0.5 * BOUNDARY_X + 0.1 * i, 0, BOUNDARY_Z - 0.1 * j, 0.01, 1.0, 0.0);
			}
	}
	time++;
	//*/
    simBox.one_tick();
    simBox.showScene();

    glutSwapBuffers();
}

/* Write a screenshot, in the PPM format, to the specified filename, in PPM format */
void saveScreenshot(int windowWidth, int windowHeight, char *filename)
{
    if (filename == NULL)
        return;

    // Allocate a picture buffer 
    Pic * in = pic_alloc(windowWidth, windowHeight, 3, NULL);

    // printf("File to save to: %s\n", filename);

    for (int i = windowHeight - 1; i >= 0; i--)
    {
        glReadPixels(0, windowHeight - i - 1, windowWidth, 1, GL_RGB, GL_UNSIGNED_BYTE,
            &in->pix[i*in->nx*in->bpp]);
    }

    ppm_write(filename, in);
    //if (ppm_write(filename, in))
    //printf("File saved Successfully\n");
    //else
    //printf("Error in Saving\n");

    pic_free(in);
}

void idle() {
    // save scene to jpg images
    string file_name = "";
    string jpg = ".ppm";
    string a = "000";
    string b = "00";
    string c = "0";
    char p[10] = {};
    if (saveImage == 1) {
        if (imageNum < 10) {
            sprintf_s(p, sizeof(p), "%d", imageNum);
            file_name = a + p + jpg;
            saveScreenshot(windowWidth, windowHeight, (char*)file_name.c_str());
        }
        else if (imageNum < 100) {
            sprintf_s(p, sizeof(p), "%d", imageNum);
            file_name = b + p + jpg;
            saveScreenshot(windowWidth, windowHeight, (char*)file_name.c_str());
        }
        else if (imageNum < 1000) {
            sprintf_s(p, sizeof(p), "%d", imageNum);
            file_name = c + p + jpg;
            saveScreenshot(windowWidth, windowHeight, (char*)file_name.c_str());
        }
        else if (imageNum < 2000) {
            sprintf_s(p, sizeof(p), "%d", imageNum);
            file_name = p + jpg;
            saveScreenshot(windowWidth, windowHeight, (char*)file_name.c_str());
        }
        else
            saveImage = false;

        imageNum++;
    }

    if (imageNum >= 2000) { // allow only 2000 snapshots
        //exit(0);
    }
    
    /*
    if (pause == 0) {
    simBox.computeAcceleration();
    simBox.computeDensity();

    simBox.showScene();
    }*/

    glutPostRedisplay();
}

void reshape(int w, int h) {
    if (h == 0)
        h = 1;

    float ratio = 1.0* w / h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glViewport(0, 0, w, h);

    gluPerspective(60, ratio, 0.01, 100);
    glMatrixMode(GL_MODELVIEW);

    windowWidth = w;
    windowHeight = h;

    glutPostRedisplay();
}

void mouseMotionDrag(int x, int y) {
    int vMouseDelta[2] = { x - g_vMousePos[0], y - g_vMousePos[1] };

    if (g_iRightMouseButton) { // handle camera rotations
        Phi -= vMouseDelta[0] * 0.01;
        Theta += vMouseDelta[1] * 0.01;

        if (Phi > 2 * PI)
            Phi -= 2 * PI;

        if (Phi < 0)
            Phi += 2 * PI;

        if (Theta > PI / 2 - 0.01) // dont let the point enter the north pole
            Theta = PI / 2 - 0.01;

        if (Theta < -PI / 2 + 0.01)
            Theta = -PI / 2 + 0.01;

        g_vMousePos[0] = x;
        g_vMousePos[1] = y;
    }
}

void mouseMotion(int x, int y) {
    g_vMousePos[0] = x;
    g_vMousePos[1] = y;
}

void mouseButton(int button, int state, int x, int y) {
    switch (button) {
    case GLUT_LEFT_BUTTON:
        g_iLeftMouseButton = (state == GLUT_DOWN);
        break;
    case GLUT_MIDDLE_BUTTON:
        g_iMiddleMouseButton = (state == GLUT_DOWN);
        break;
    case GLUT_RIGHT_BUTTON:
        g_iRightMouseButton = (state == GLUT_DOWN);
        break;
    }

    g_vMousePos[0] = x;
    g_vMousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        //exit(0);
        break;
    case 'p':
    case 'P':
        pause = 1 - pause;
        break;
    case ' ':
        saveImage = 1 - saveImage;
        break;
	case 'W':
	case 'w':
		R -= 0.1;
		break;
	case 'S':
	case 's':
		R += 0.1;
		break;
    }
}

// set fragment shader and vertex shader here
void setShaders() {

}

int main(int argc, char ** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    windowWidth = windowHeight = 600;
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Fluid Simulation");

    GLenum err = glewInit();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);

    /* callback for mouse drags */
    glutMotionFunc(mouseMotionDrag);

    /* callback for mouse movement */
    glutPassiveMotionFunc(mouseMotion);

    /* callback for mouse button changes */
    glutMouseFunc(mouseButton);

    /* register for keyboard events */
    glutKeyboardFunc(keyboardFunc);

	cudaSetDevice(0);

    myinit(); // initialization
    
    glutMainLoop();

    return 0;
}
