// 假设你有函数来处理用户 API 请求

export const loginUser = async (email, password) => {
    const response = await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      headers: { 'Content-Type': 'application/json' },
    });
    return await response.json();
  };
  
  export const getUserInfo = async (userId) => {
    const response = await fetch(`/api/users/${userId}`);
    return await response.json();
  };
  
  // 其他用户相关操作...
  