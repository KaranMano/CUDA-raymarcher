#include "list.h"
	
__host__ __device__ List::List() : head(nullptr), size(0) {}
__host__ __device__ List::~List() {
	Node *curr = head;
	while (curr != nullptr) {
		Node *next = curr->next;
		delete curr;
		curr = next; 
	}
}

__host__ __device__ void List::push(Node *curr) {
	if (curr == nullptr)
		return;
	curr->next = head;
	head = curr;
	size++;
}
__host__ __device__ Object* List::pop() {
	if (head == nullptr)
		return nullptr;
	Node *curr = head->next;
	delete head;
	head = curr;
	size--;
}
