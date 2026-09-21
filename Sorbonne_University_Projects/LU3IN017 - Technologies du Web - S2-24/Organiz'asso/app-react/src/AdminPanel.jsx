import React, { useState } from 'react';
import AdminMenu from './AdminMenu';
import UserValidation from './UserValidation';
import AdminFeed from './AdminFeed';

function AdminPanel() {
  const [selectedSection, setSelectedSection] = useState('adminFeed');

  return (
    <div>
      <AdminMenu setSelectedSection={setSelectedSection} />
      {selectedSection === 'userValidation' && <UserValidation />}
      {selectedSection === 'adminFeed' && <AdminFeed />}
    </div>
  );
}

export default AdminPanel;
