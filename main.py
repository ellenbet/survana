from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class TodoItem(BaseModel):
    id: int
    content: str


class TodoItemCreate(BaseModel):
    content: str


todos: list[TodoItem] = []
id_counter: int = 1


@app.post("/todos", response_model=TodoItem)
async def create_todo(item: TodoItemCreate) -> TodoItem:
    global id_counter
    new_todo: TodoItem = TodoItem(id=id_counter, content=item.content)
    todos.append(new_todo)
    id_counter += 1
    return new_todo


@app.get("/todos", response_model=list[TodoItem])
async def read_todos() -> list[TodoItem]:
    return todos


@app.delete("/todos{todo_id}")
async def delete_todo(todo_id: int) -> dict[str, str] | None:
    for index, todo in enumerate(todos):
        if todo.id == todo_id:
            todos.pop(index)
            return {"message": "todo deleted"}
    raise HTTPException(status_code=404, detail="todo not found")
