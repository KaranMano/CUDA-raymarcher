#pragma once
#include "object.h"

class Node {
	public:
	Object obj;
	Node *next;
};

class List{
	public:
	Node *head;
	int size;
	
	__host__ __device__ List();
	__host__ __device__ ~List();
	
	__host__ __device__ void push(Node *curr);
	__host__ __device__ Object* pop();
};