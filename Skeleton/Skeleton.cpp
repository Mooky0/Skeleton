//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tóth Gábor
// Neptun : F041OM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

//a poincare.cpp demo file-ból masolt osztaly.
class ImmediateModeRenderer2D : public GPUProgram {
	const char* const vertexSource = R"(
		#version 330
		precision highp float;
		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		void main() { gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); }	
	)";

	const char* const fragmentSource = R"(
		#version 330
		precision highp float;
		uniform vec3 color;
		out vec4 fragmentColor;	

		void main() { fragmentColor = vec4(color, 1); }
	)";

	unsigned int vao, vbo;

	int Prev(std::vector<vec2> polygon, int i) { return i > 0 ? i - 1 : polygon.size() - 1; }
	int Next(std::vector<vec2> polygon, int i) { return i < polygon.size() - 1 ? i + 1 : 0; }

	bool intersect(vec2 p1, vec2 p2, vec2 q1, vec2 q2) {
		return (dot(cross(p2 - p1, q1 - p1), cross(p2 - p1, q2 - p1)) < 0 &&
			dot(cross(q2 - q1, p1 - q1), cross(q2 - q1, p2 - q1)) < 0);
	}

	bool isEar(const std::vector<vec2>& polygon, int ear) {
		int d1 = Prev(polygon, ear), d2 = Next(polygon, ear);
		vec2 diag1 = polygon[d1], diag2 = polygon[d2];
		for (int e1 = 0; e1 < polygon.size(); e1++) {
			int e2 = Next(polygon, e1);
			vec2 edge1 = polygon[e1], edge2 = polygon[e2];
			if (d1 == e1 || d2 == e1 || d1 == e2 || d2 == e2) continue;
			if (intersect(diag1, diag2, edge1, edge2)) return false;
		}
		vec2 center = (diag1 + diag2) / 2.0f;
		vec2 infinity(2.0f, center.y);
		int nIntersect = 0;
		for (int e1 = 0; e1 < polygon.size(); e1++) {
			int e2 = Next(polygon, e1);
			vec2 edge1 = polygon[e1], edge2 = polygon[e2];
			if (intersect(center, infinity, edge1, edge2)) nIntersect++;
		}
		return (nIntersect & 1 == 1);
	}

	void Triangulate(const std::vector<vec2>& polygon, std::vector<vec2>& triangles) {
		if (polygon.size() == 3) {
			triangles.insert(triangles.end(), polygon.begin(), polygon.begin() + 2);
			return;
		}

		std::vector<vec2> newPolygon;
		for (int i = 0; i < polygon.size(); i++) {
			if (isEar(polygon, i)) {
				triangles.push_back(polygon[Prev(polygon, i)]);
				triangles.push_back(polygon[i]);
				triangles.push_back(polygon[Next(polygon, i)]);
				newPolygon.insert(newPolygon.end(), polygon.begin() + i + 1, polygon.end());
				break;
			}
			else newPolygon.push_back(polygon[i]);
		}
		Triangulate(newPolygon, triangles);
	}

	std::vector<vec2> Consolidate(const std::vector<vec2> polygon) {
		const float pixelThreshold = 0.01f;
		vec2 prev = polygon[0];
		std::vector<vec2> consolidatedPolygon = { prev };
		for (auto v : polygon) {
			if (length(v - prev) > pixelThreshold) {
				consolidatedPolygon.push_back(v);
				prev = v;
			}
		}
		if (consolidatedPolygon.size() > 3) {
			if (length(consolidatedPolygon.back() - consolidatedPolygon.front()) < pixelThreshold) consolidatedPolygon.pop_back();
		}
		return consolidatedPolygon;
	}

public:
	ImmediateModeRenderer2D() {
		glViewport(0, 0, windowWidth, windowHeight);
		glLineWidth(2.0f); glPointSize(10.0f);

		create(vertexSource, fragmentSource, "outColor");
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
	}

	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_DYNAMIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}

	void DrawPolygon(std::vector<vec2> vertices, vec3 color) {
		std::vector<vec2> triangles;
		Triangulate(Consolidate(vertices), triangles);
		DrawGPU(GL_TRIANGLES, triangles, color);
	}

	~ImmediateModeRenderer2D() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

vec3 normalVector(vec3 point, vec3 ir) {
	vec3 v = vec3(ir.x, ir.y, -ir.z);
	vec3 p = vec3(point.x, point.y, -point.z);
	return cross(v, p);
}

double hyperbolicDot(vec3 v1, vec3 v2) {
	return v1.x * v2.x + v1.y * v2.y - v1.z * v2.z;
}

double hyperboliclenght(vec3 v) {
	return sqrtf(hyperbolicDot(v, v));
}

const int nTesselatedVertices = 30;

vec3 centralInverseProjectionVector(vec3 p, vec3 v) {
	float lambda = hyperbolicDot(p, v);
	return v + lambda * p;
}

vec3 hyperbolicNormalize(vec3 v) {
	return v / sqrtf(fabs(hyperbolicDot(v, v)));
}

vec3 FixVector(vec3 point, vec3 vector) {
	vec3 result = vector;
	if (-hyperbolicDot(point, vector) > 1.e-6) {
		result = result + (point * (-hyperbolicDot(point, vector) / hyperbolicDot(point, point)));
	}
	return result;
}

vec3 rotateVector(vec3 point, vec3 vector, float angle) {
	vec3 ret = hyperbolicNormalize(vector);
	return hyperbolicNormalize(FixVector(point, vec3(ret * cosf(angle) + normalVector(point, ret) * sinf(angle))));
}


vec3 normalInvertedProjection(vec3 v) {
	vec3 v2 = vec3(v.x, v.y, v.z);
	v2.z = sqrt(v2.x * v2.x + v2.y * v2.y + 1);
	return v2;
}

vec2 poincareProjection(vec3 pos) {
	return vec2(pos.x / (pos.z + 1), pos.y / (pos.z + 1));
}

vec3 hyCross(vec3 v, vec3 q) {
	return cross(vec3(v.x, v.y, -v.z), vec3(q.x, q.y, -q.z));
}

double hyDist(vec3 p, vec3 q) {
	return acoshf(-hyperbolicDot(p, q));
}

vec3 hyDir(vec3 p, vec3 q) {
	return vec3((q - p * coshf(hyDist(p, q))) / sinhf(hyDist(p, q)));
}

ImmediateModeRenderer2D* renderer;

class Hami {
public:
	vec3 pos;
	vec3 heading;
	std::vector<vec2> trail;
	double r = 1.0f / 5.0f;

	Hami() {
		vec3 pos = vec3(0.0f, 0.0f, 1.0f);
		vec3 heading = hyperbolicNormalize(vec3(0.0f, 1, 0));
	}

	Hami(float x, float y, float z, float hx, float hy, float hz) {
		pos = hyperbolicNormalize(vec3(x, y, z));
		heading = FixVector(pos, vec3(hx, hy, hz));
		
	}

	void draw(vec3 color, long t, Hami lookAt) {
		vec3 direction;
		if (hyDist(pos, vec3(0, 0, 1)) <= 0.0001) {
			direction = vec3(1, 0, 0);
		}
		else {
			direction = hyDir(pos, vec3(0, 0, 1));
		}
		vec3 prep = hyperbolicNormalize(hyCross(pos, direction));
		std::vector<vec3> circlePoints;
		std::vector<vec2> circlePointsProj;
		std::vector<vec2> mouth;
		std::vector<vec2> rightEye;
		std::vector<vec2> leftEye;
		std::vector<vec2> rightPupil;
		std::vector<vec2> leftPupil;

		vec3 dirToHami = hyperbolicNormalize(hyDir(pos, lookAt.pos)) *r;

		for (int i = 0; i < nTesselatedVertices; ++i) {
			float phi = 2 * M_PI * i / nTesselatedVertices;
			vec3 dir = hyperbolicNormalize(direction * cosf(phi) + prep * sinf(phi));
			vec3 point = pos * coshf(r) + dir * sinhf(r);
			circlePoints.push_back(hyperbolicNormalize(point));
			circlePointsProj.push_back(poincareProjection(circlePoints[i]));
		}

		for (int i = 0; i < nTesselatedVertices; i++) {
			float phi = i * 2.0f * M_PI / nTesselatedVertices;
			vec3 dir = hyperbolicNormalize(direction * cosf(phi) + prep * sinf(phi));
			vec3 point = (pos + hyperbolicNormalize(heading) * r) * coshf(r /5 * sinf(t /500.0f)) + dir * sinhf(r /5 * sinf(t/500.0f));
			mouth.push_back(poincareProjection(hyperbolicNormalize(point)));
		}

		for (int i = 0; i < nTesselatedVertices; i++) {
			float phi = i * 2.0f * M_PI / nTesselatedVertices;
			vec3 dir = hyperbolicNormalize(direction * cosf(phi) + prep * sinf(phi));
			
			vec3 rightEyeCenter = pos + hyperbolicNormalize(rotateVector(pos, heading, -0.5f)) * r;
			vec3 point = rightEyeCenter * coshf(r / 3) + dir * sinhf(r / 3);
			rightEye.push_back(poincareProjection(hyperbolicNormalize(point)));

			point = normalInvertedProjection(rightEyeCenter * coshf(r) + dirToHami * sinhf(r)) * coshf(r / 7) + dir * sinhf(r / 7);
			rightPupil.push_back(poincareProjection(hyperbolicNormalize(point)));


			vec3 leftEyeCenter = pos + hyperbolicNormalize(rotateVector(pos, heading, 0.5f)) * r;
			point = leftEyeCenter * coshf(r / 3) + dir * sinhf(r / 3);
			leftEye.push_back(poincareProjection(hyperbolicNormalize(point)));
			
			point = normalInvertedProjection(leftEyeCenter * coshf(r) + dirToHami * sinhf(r)) * coshf(r / 7) + dir * sinhf(r / 7);
			leftPupil.push_back(poincareProjection(hyperbolicNormalize(point)));
			
		}
		
		renderer->DrawGPU(GL_LINE_STRIP, trail, vec3(1, 1, 1));
		renderer->DrawGPU(GL_TRIANGLE_FAN, circlePointsProj, color);
		renderer->DrawGPU(GL_TRIANGLE_FAN, rightEye, vec3(1, 1, 1));
		renderer->DrawGPU(GL_TRIANGLE_FAN, leftEye, vec3(1, 1, 1));
		renderer->DrawGPU(GL_TRIANGLE_FAN, rightPupil, vec3(0, 0, 1));
		renderer->DrawGPU(GL_TRIANGLE_FAN, leftPupil, vec3(0, 0, 1));
		renderer->DrawGPU(GL_TRIANGLE_FAN, mouth, vec3(0, 0, 0));
	}

	vec2 getPoincare() {
		return vec2(pos.x/(pos.z+1), pos.y/(pos.z+1));
	}

	vec2 planeHeading() {
		return normalize(vec2(heading.x / (heading.z + 1), heading.y / (heading.z + 1)));
	}

	void forward(double t) {
		pos = normalInvertedProjection(pos);
		heading = FixVector(pos, heading);
		vec3 oldPos = pos;
		pos = coshf(t) * pos + sinhf(t) * heading;
		pos = normalInvertedProjection(pos);
		
		heading = sinhf(t) * oldPos + heading * coshf(t);
		heading = FixVector(pos, heading);
		heading = hyperbolicNormalize(heading);
		trail.push_back(getPoincare());
	}
	void rightTurn(double t) {
		heading = FixVector(pos, heading);
		heading = rotateVector(pos, heading, t);
		heading = hyperbolicNormalize(heading);
	}
	void leftTurn(double t) {
		heading = FixVector(pos, heading);
		heading = rotateVector(pos, heading, -t);
		heading = hyperbolicNormalize(heading);
	}
};

const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram;
unsigned int vao;
float vertecies[6];
std::vector<vec2> circlePoints;
Hami pirosHami = Hami(0.0f, 0.0f, 1.0f, 0.0f, 1, 0);
Hami zoldHami = Hami(2.0f, 2.0f, 3.0f, 1, 1, 4.0f/3);
bool e;
bool f;
bool s;
long lastFrame = 0;

void onInitialization() {
	renderer = new ImmediateModeRenderer2D();
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	unsigned int vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glLineWidth(3);

	for (int i = 0; i < nTesselatedVertices; i++) {
		float phi = i * 2.0f * M_PI / nTesselatedVertices;
		circlePoints.push_back(vec2(cosf(phi), sinf(phi)));
	}

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,
		2, GL_FLOAT, GL_FALSE,
		0, NULL);


	renderer->create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	glClearColor(0.5f, 0.5f, 0.5f, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	int location = glGetUniformLocation(renderer->getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f);
	renderer->DrawGPU(GL_TRIANGLE_FAN, circlePoints, vec3(0, 0, 0));
	zoldHami.draw(vec3(0, 1, 0), time, pirosHami);
	pirosHami.draw(vec3(1, 0, 0), time, zoldHami);

	float MVPtransf[4][4] = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(renderer->getId(), "MVP");
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0 , 3);

	printf("pos: (%.2f, %.2f, %.2f)\t", pirosHami.pos.x, pirosHami.pos.y, pirosHami.pos.z);
	printf("@sik: (%.2f, %.2f)\n", pirosHami.getPoincare().x, pirosHami.getPoincare().y);
	printf("heading: (%.2f, %.2f, %.2f)", pirosHami.heading.x, pirosHami.heading.y, pirosHami.heading.z);
	printf("valid: %.2f, lorentz: %.2f\n", hyperbolicDot(pirosHami.pos, pirosHami.pos), hyperbolicDot(pirosHami.pos, pirosHami.heading));
	printf("dist: %f.4\n", hyDist(vec3(0, 0, 1), pirosHami.pos));
	printf("=================\n");

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	printf("key pressed: %c\n", key);
	switch (key)
	{
	case('e'):
		e = true;
		break;
	case('f'):
		f = true;
		break;
	case('s'):
		s = true;
		break;
	default:
		break;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
	printf("keyUp: %c\n", key);
	switch (key)
	{
	case('e'):
		e = false;
		break;
	case('f'):
		f = false;
		break;
	case('s'):
		s = false;
		break;
	default:
		break;
	}
}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	if (time - lastFrame > 30) {
		for (int i = 0; i < (time - lastFrame) / 30; i++) {
			if (e) {
				pirosHami.forward(0.05f);
			}
			if (f) {
				pirosHami.rightTurn(M_PI / 10);
			}
			if (s) {
				pirosHami.leftTurn(M_PI / 10);
			}
			zoldHami.forward(0.1f);
			zoldHami.rightTurn(M_PI / 10);
		}
		glutPostRedisplay();
		lastFrame = time;
	}
}
