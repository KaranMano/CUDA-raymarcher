//#include "bounding-box.h"
//
// BoundingBox::BoundingBox(const Vector &_center, float _radius, Material* _mat, bool _volume) :
//	center(_center), radius(_radius), Object(_mat, _volume)
//{
//	std::cout << "center: " << center.x << " " << center.y << " " << center.z << "\n";
//	std::cout << "radius: " << radius << "\n";
//	std::cout << "color: " << material->color.x << " " << material->color.y << " " << material->color.z << "\n";
//}
//  thrust::pair<bool, float> BoundingBox::intersect(const Ray& ray) const {
//	return { false, -1.0f };
//}
//
//  Vector BoundingBox::getNormal(const Vector &point) const {
//	return { 0.0f, 0.0f, 0.0f};
//}