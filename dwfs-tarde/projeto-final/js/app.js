document.addEventListener('DOMContentLoaded', () => {
    const taskInput = document.getElementById('task-input');
    const addBtn = document.getElementById('add-btn');
    const taskList = document.getElementById('task-list');

    function addTask() {
        const taskText = taskInput.value.trim();
        
        if (!taskText) {
            alert('Por favor, insira uma tarefa antes de adicionar!');
            return;
        }

        const li = document.createElement('li');
        
        const span = document.createElement('span');
        span.textContent = taskText;
        li.appendChild(span);

        const removeBtn = document.createElement('button');
        removeBtn.textContent = 'Remover';
        removeBtn.className = 'remove-btn';
        removeBtn.addEventListener('click', () => {
            li.remove();
        });
        
        li.appendChild(removeBtn);
        taskList.appendChild(li);

        taskInput.value = '';
        taskInput.focus();
    }

    if (addBtn) {
        addBtn.addEventListener('click', addTask);
    }

    if (taskInput) {
        taskInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addTask();
            }
        });
    }
});